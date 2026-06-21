"""Portfolio Monte Carlo Risk Simulator

Run as a script or import `PortfolioSimulator`.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float


class PortfolioSimulator:
    """Monte Carlo portfolio risk simulator using GBM and Cholesky for correlation.

    Attributes:
        tickers: list of ticker symbols
        start_date, end_date: data window
        n_sims: number of Monte Carlo simulations
        horizon_days: simulation horizon in trading days
        initial_value: starting portfolio value in currency units
        weights: portfolio weights (if None, equal weights used)
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        n_sims: int = 10000,
        horizon_days: int = 252,
        initial_value: float = 1_000_000.0,
        weights: Optional[List[float]] = None,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.n_sims = n_sims
        self.horizon_days = horizon_days
        self.initial_value = float(initial_value)
        self.weights = np.array(weights) if weights is not None else None

        self.prices = None
        self.returns = None
        self.mu = None
        self.sigma = None
        self.corr = None

    def fetch_data(self) -> pd.DataFrame:
        """Fetches adjusted close prices using yfinance.

        Handles both single-level and MultiIndex returns from yfinance.download.
        """
        logger.info("Fetching data for %s from %s to %s", self.tickers, self.start_date, self.end_date)
        # auto_adjust=True returns adjusted prices so we can work with 'Close'
        raw = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)

        # Robust handling of different return shapes from yfinance
        if isinstance(raw, pd.DataFrame):
            try:
                # prefer adjusted 'Close' (auto_adjust=True) or 'Adj Close' if present
                if "Close" in raw.columns:
                    data = raw["Close"]
                elif "Adj Close" in raw.columns:
                    data = raw["Adj Close"]
                elif isinstance(raw.columns, pd.MultiIndex) and ("Close" in raw.columns.get_level_values(0) or "Adj Close" in raw.columns.get_level_values(0)):
                    data = raw.get("Close") or raw.get("Adj Close")
                else:
                    # fallback: raw likely already has prices
                    data = raw
            except Exception:
                data = raw
        elif isinstance(raw, pd.Series):
            data = raw.to_frame()
        else:
            raise RuntimeError("Unexpected data format returned from yfinance")

        data = data.dropna()
        if data.empty:
            raise RuntimeError("No price data fetched; check tickers/date range and network.")
        self.prices = data
        self.returns = data.pct_change().dropna()
        logger.info("Fetched %d rows of price data", len(data))
        return data

    def estimate_parameters(self) -> None:
        """Estimate drift (mu), vol (sigma), and correlation matrix from returns."""
        if self.returns is None:
            raise RuntimeError("Call fetch_data() before estimating parameters.")

        # Annualize mean and std (assume 252 trading days)
        mu_daily = self.returns.mean()
        sigma_daily = self.returns.std()

        self.mu = mu_daily * 252
        self.sigma = sigma_daily * np.sqrt(252)
        self.corr = self.returns.corr()
        logger.info("Estimated mu (annual): %s", self.mu.to_dict())
        logger.info("Estimated sigma (annual): %s", self.sigma.to_dict())

    def run_simulations(self) -> np.ndarray:
        """Run Monte Carlo simulations for asset final prices at horizon using multivariate normals.

        Returns:
            final_prices: ndarray shape (n_sims, n_assets)
        """
        if self.prices is None:
            raise RuntimeError("Call fetch_data() before running simulations.")
        if self.mu is None or self.sigma is None or self.corr is None:
            self.estimate_parameters()

        assets = list(self.prices.columns)
        n_assets = len(assets)
        S0 = self.prices.iloc[-1].values.astype(float)

        # Annual horizon scaling
        T = self.horizon_days / 252.0

        mu_vec = np.array(self.mu[assets], dtype=float)
        sigma_vec = np.array(self.sigma[assets], dtype=float)

        # Covariance for aggregated returns over horizon T
        cov_matrix = np.outer(sigma_vec, sigma_vec) * self.corr.values * T

        # Mean vector for log-returns over T
        mean_log = (mu_vec - 0.5 * sigma_vec ** 2) * T

        logger.info("Simulating %d paths for %d assets over T=%.3f years", self.n_sims, n_assets, T)

        # Draw multivariate normals
        rng = np.random.default_rng()
        draws = rng.multivariate_normal(mean=np.zeros(n_assets), cov=cov_matrix, size=self.n_sims)

        # Add mean drift
        log_returns = draws + mean_log

        final_prices = S0 * np.exp(log_returns)
        return final_prices

    def portfolio_metrics(self, final_prices: np.ndarray) -> VaRResult:
        """Compute portfolio VaR and CVaR at 95% and 99%.

        final_prices: shape (n_sims, n_assets)
        """
        n_assets = final_prices.shape[1]

        if self.weights is None:
            weights = np.repeat(1.0 / n_assets, n_assets)
        else:
            weights = np.array(self.weights, dtype=float)
            weights = weights / weights.sum()

        # portfolio final values
        final_portfolio_values = (final_prices * weights).sum(axis=1) * (self.initial_value / (self.prices.iloc[-1].values * weights).sum())

        losses = np.maximum(0.0, self.initial_value - final_portfolio_values)

        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)

        cvar_95 = losses[losses >= var_95].mean() if np.any(losses >= var_95) else var_95
        cvar_99 = losses[losses >= var_99].mean() if np.any(losses >= var_99) else var_99

        logger.info("VaR95=%.2f, VaR99=%.2f, CVaR95=%.2f, CVaR99=%.2f", var_95, var_99, cvar_95, cvar_99)

        return VaRResult(var_95=var_95, var_99=var_99, cvar_95=cvar_95, cvar_99=cvar_99)

    def plot_and_save(self, losses: np.ndarray, var_result: VaRResult, out_path: Optional[Path] = None) -> Path:
        """Create histogram of losses and annotate VaR/CVaR, then save to out_path."""
        if out_path is None:
            out_path = Path(__file__).resolve().parent / "portfolio_risk_chart.png"

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(losses, bins=100, color="#3b7dd8", alpha=0.8)

        ax.axvline(var_result.var_95, color="orange", linestyle="--", linewidth=2, label=f"VaR 95% = {var_result.var_95:,.0f}")
        ax.axvline(var_result.var_99, color="red", linestyle="--", linewidth=2, label=f"VaR 99% = {var_result.var_99:,.0f}")

        ax.axvline(var_result.cvar_95, color="orange", linestyle=":", linewidth=2, label=f"CVaR 95% = {var_result.cvar_95:,.0f}")
        ax.axvline(var_result.cvar_99, color="red", linestyle=":", linewidth=2, label=f"CVaR 99% = {var_result.cvar_99:,.0f}")

        ax.set_title("Portfolio Loss Distribution and Risk Metrics")
        ax.set_xlabel("Loss (currency units)")
        ax.set_ylabel("Frequency")
        ax.legend()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info("Saved chart to %s", out_path)
        return out_path

    def run_and_save(self) -> VaRResult:
        final_prices = self.run_simulations()
        if self.weights is None:
            n_assets = final_prices.shape[1]
            weights = np.repeat(1.0 / n_assets, n_assets)
        else:
            weights = np.array(self.weights, dtype=float)
            weights = weights / weights.sum()

        final_portfolio_values = (final_prices * weights).sum(axis=1) * (self.initial_value / (self.prices.iloc[-1].values * weights).sum())
        losses = np.maximum(0.0, self.initial_value - final_portfolio_values)

        var_result = self.portfolio_metrics(final_prices)
        out_path = Path(__file__).resolve().parent / "portfolio_risk_chart.png"
        self.plot_and_save(losses, var_result, out_path)
        return var_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo Portfolio Risk Simulator")
    p.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA", help="Comma-separated tickers")
    p.add_argument("--start", type=str, default=(pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d"), help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    p.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations")
    p.add_argument("--horizon", type=int, default=252, help="Horizon in trading days (default 252)")
    p.add_argument("--initial", type=float, default=1_000_000.0, help="Initial portfolio value")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    sim = PortfolioSimulator(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        n_sims=args.sims,
        horizon_days=args.horizon,
        initial_value=args.initial,
    )

    sim.fetch_data()
    sim.estimate_parameters()
    var_res = sim.run_and_save()

    print(f"VaR95: {var_res.var_95:,.2f}")
    print(f"VaR99: {var_res.var_99:,.2f}")
    print(f"CVaR95: {var_res.cvar_95:,.2f}")
    print(f"CVaR99: {var_res.cvar_99:,.2f}")


if __name__ == "__main__":
    main()
