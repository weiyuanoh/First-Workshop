Credit to EDHEC Risk Management for the libraries, I only edited and added new functions I deem needed for this project

# Portfolio 
A portfolio study of the returns if we select 10 stocks from the S&P 500 index. A backtest of the portfolio returns is carried out and plotted according to weights obtained from 
{ 'EW' : Equal Weighted,
  'GMV-SAMPLE' : Global Minimum Variance using a standard Covariance estimate of historial returns,
  'GMV-CC' : Global Minimum Variance using a covariance matrix estimated by the Elton/Gruber Constant Correlation model,
  'GMV-Shrink 0.5' : Global Minimum Variance using a Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators,\
  'MSR' : The Max Sharpe Ratio 
  }



# Portfolio 2
Similar Analysis with Portfolio, but using Singapore stocks here.


# ff_analysis 
A study of the portfolio of US Stocks using Fama-French Models, using maching learning techniques to predict factor tilt of certain stocks in the portfolio. 
