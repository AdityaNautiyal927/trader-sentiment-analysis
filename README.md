Trader Performance vs Market Sentiment Analysis

This analysis explores how trader performance varies across market sentiment regimes using 211,224 trades and 2,644 daily sentiment observations.

The datasets were merged on daily timestamps after cleaning and date alignment. Missing values were negligible and duplicates were removed.

Key findings:

Traders achieve the highest mean PnL (67.89) and win rate (46.5%) during Extreme Greed.

Risk-adjusted returns are strongest in Extreme Greed (0.0885), indicating efficient capital deployment.

During Fear, average trade size increases significantly (7,816 USD), but efficiency declines.

Neutral markets produce stable and consistent performance with favorable risk-adjusted metrics.

Segmentation analysis shows trading volume is dominated by frequent traders, and a majority of accounts demonstrate relatively consistent return patterns.

Strategically, traders should consider scaling exposure based on sentiment regimes and reducing risk during fear-driven volatility.

Strategy Recommendations:

Increase exposure during extreme greed regimes while controlling downside risk.
Reduce leverage during extreme fear periods dur to lower risk-adjusted performance.
Prioritize capital allocation toward consistant traders.
Improve model performance using rolling sentiment and  volatility features.

##Setup and Execution:
1. Create virtual envirenment: python -m venv venv
2. Activate: venv\Scripts\activate
3. Install dependancies: pip install -r requirements.txt
4. Run analysis: python analysis.py