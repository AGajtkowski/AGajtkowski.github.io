---
title: "Constructing Underemployment Index using the Labour Force Survey. Unit toot tests, time-series regression and the Granger Causality tests"
date: 2019-03-25
tags: [LFS, time-series, STATA, EViews]
header:
  image: "images/background.jpg"
excerpt: "LFS, time-series, STATA"
---
I compute underemployment index and conduct the time series analysis. Bell and Blanchflower (2013) estimated underemployment index for the UK using individual data provided in the Labour Force Survey (LFS), accessed on the UK Dataservice website . Quarterly survey also provides necessary data for estimation of the ONS unemployment rate and underemployment rate. Each LFS includes approximetaly 130,000 observations representative for the whole UK.

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/pic1.png"></a>
</figure>

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/pic2.png"></a>
</figure>

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/pic3.png"></a>
</figure>

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/pic4.png"></a>
</figure>

Bell and Blanchflower use LFS to calculate unemployment rate using weights provided by the ONS. Authors include those who are employed, self-employed, family workers and those on government schemes to calculate total employment and average working hours. Researchers used X-13ARIMA-SEATS method of seasonal adjustment. They show that seasonal adjustment does not make any significant difference in estimate of underemployment index. In other words, authors report “quarterly seasonal effects to be small” (Bell and Blanchflower, 2013). Bell and Blanchflower follow ONS definition of underemployment and overemployment. For underemployment, authors do not take into consideration those aged 16-19 working 40 or more hours, and those aged over 18 working 48 hours or more. For underemployment, subjects aged 16 – 18 working 15 hours or less are not considered. Subjects aged 18 and more working 20 hours and less are not taken into consideration as well (Bell and Blanchflower, 2013).

I use ONS estimates of average weekly hours worked, number of people in employment and number of unemployed people. All data are seasonally adjusted by the ONS. I use ONS data as I want to directly compare ONS unemployment rate with Bell and Blanchflower underemployment index. Changes in methodology, or different method of seasonal adjustment (as proposed by Bell and Blanchflower) could distort direct comparison. I perform quality assurance, by comparing ONS unemployment rate with unemployment rate expressed as a ratio of hours I computed. Both rates are identical. There are four exceptions associated with approximation rules, as the ONS round numbers such as 5.05  down to 5.0, while it should be round up to 5.1. It does not significantly affect underemployment index calculated by me.

In order to calculate underemployment index, I estimate sum of underemployed and overemployed hours and transform it from sample data to population estimates. I do not use seasonal adjustment, as I have already included 3 variables; unemployed, employed, average hours worked, which are seasonally adjusted by the ONS. Bell and Blanchfower (2013) note that seasonal adjustment does not change estimates significantly. To calculate sum of underemployed/overemployed hours I consider data based on following questions in the Labour Force Survey (Ons.gov.uk, 2018):

252 UNDEMP
Would you prefer to work longer hours at your current basic rates – that is, not overtime or enhanced pay rates – if you were given the opportunity?
1	yes
2	no
Applies if respondent is not looking for a different or additional job.


253 UNDHRS
How many extra hours, in addition to those you usually work, would you like to work each week?
97 = 97 or more
99 = do not know or refusal
Applies if respondent would prefer to work longer hours.


262 	
Would you prefer to work shorter hours than at present in your current job?
1	yes
2	no
Applies DIFFJOB=2 (not looking for another job)
and UNDEMP=2 (does not want job with more hours)

264 OVHRS
How many fever hours would you like to work in that/your current job?
97 = 97 or more
99 = don’t know or refusal
applies if LESPAY=1 (work shorter hours for less pay)
or if LESPAY3 = 1 (work shorter hours in current job for less pay)

I compute underemployed/overemployed hours for each quarter using the LFS. For underemployed hours, firstly I detect all underemployed people. Then, I multiply amount of additionally desired hours given underemployed individual reports by weight assigned to this observation (by the ONS). In the end, I sum all additionally desired hours within whole sample at a given quarter. I repeat symmetric procedure for all overemployed people in each quarter. Those estimate return number of underemployed hours and overemployed hours for each quarter from 2000 q2 to 2017 q4.

Weight assigned to each observation within LFS is a numerical value used to transform sample data to data representative for whole population. I use historical weights (PWT07 to PWT 16) for years 2000 to 2016 to transform sample estimates into population estimates. I use weights PWT17 for data starting in q3 2016, as they are included in the LFS (Table 4) available for researchers.

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/pic5.png"></a>
</figure>

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/pic6.png"></a>
</figure>

Both underemployment index and unemployment rate continue to fall in years 2014 – 2017 (Graph 2). They reach its pre-downturn level in 2017 q1 and later drop to even lower levels. In 2017 q4 the difference between underemployment index and unemployment rate is just 0.31 percent point. Initially, it suggests that the British labour market comes back to its pre-downturn equilibrium in terms of its overall level, as well as underemploymed/overemployed hours. However, when I look at the Graph 3, which shows the sum of underemployed and overemployed hours, it is visible that underemployment index and unemployment rate are similar, because underemployed hours and overemployed hours cancel each other out. Aggregate level of unutilised hours and not desired additional hours is actually historically high. The overall sum of mismatched hours increased in 2007 q3 and remained at the same high level. Underemployed hours remain at the high post-downturn level, while desired reductions increased since 2013 q3. This situation highlights, that researchers need to be aware of limitations of the Bell and Blanchflower underemployment index, because it does not necessarily capture total excess capacity on a labour market. Bell and Blanchflower (2014) argued that the index gives a broader estimate of the extent of unused capacity within economy. This could be true for the post-downturn period they analysed, when underemployed hours greatly exceeded overemployed hours. However, I would always suggest reporting underemployed/overemployed hours next to underemployment index, to give more insight into the condition of a labour market.

4 Time Series Analysis

To analyse the relationship, I use three variables: unemployment rate, underemployment index and real wage. I base model specification on Gregg and Fernandez-Salgado (2014) paper. Authors use yearly data in their publication. I use quarterly data, as underemployment index is only available between years 2000 and 2017. Using yearly data would result in 17 observations. According to Box and Tiao (1975) this is not sufficient amount of observations for time series analysis.

4.1.1 Real Wage

To calculate real wage, I use the ONS average weekly earnings (total pay – Table 19 in appendices). To obtain quarterly data, I take three months’ average. Gregg and Fernandez-Salgado (2014) use yearly New Earnings Survey (NES) and Annual Survey of Hours and Earnings (ASHE) data sets. These data are based on larger population sample, accounting for approximately 1% of all workers. Because these are yearly data, I need to use smaller sample ONS data.

In my work, I use CPI to deflate nominal wage (Table 19). Gregg and Fernandez-Salgado (2014) test robustness using different deflators. They include RPIJ and GDP deflators, which are no longer produced by the ONS. CPIH deflator, currently recommended by the ONS (Ons.gov.uk, 2018), is available just from 2005. In my work I can use either CPI, or RPI deflator. I use CPI, as it meets international standards (contrary to RPI) and it is less volatile than RPI (Graph 4).

Graph 4 Consumer Price Index (CPI) and Retail Price Index (RPI), 2000 q2 to 2017 q4, Index 2015=100. Source: ONS

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture1.png"></a>
</figure>

4.2 Stationarity

Time series analysis requires stationary variables. The main reason why I need to test for the presence of a unit root is the spurious regression discussed by Granger and Newbold (1974) and Phillips. In case of presence of a unit root in either real wage, unemployment rate, or underemployment index, even if those variables are independent, estimators and tests might suggest that there is a relationship between them (Hamilton, 1994).

4.2.1 Pre-test analysis and Augmented Dickey Fuller Tests

Between q2 2000 and q3 2008 real wage resembles trend stationary model (Graph 5). Real wage growths over time with fluctuations around the trend (as in Keynesian theories). The effect of unexpected shock, the 2008 downturn, is persistent. Level of wages stagnates after this shock and does not come back to the pre-downturn trend. This may suggest that real wage follows a unit root model (such as in real business cycle models) – it has a long memory.

Graph 5 Real Wage (W). British Pound, q2 2000 to q4 2017, Deflated by CPI 100 = 2015. Source: ONS
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture2.png"></a>
</figure>

Graph 6 Underemployment Index and Unemployment Rate. q2 2000 to q4 2017. Source: LFS

<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture3.png"></a>
</figure>
Now let us look at the Graph 2 from page 14 again, this time in context of stationarity (Graph 6). Unemployment rate and underemployment index follow a similar path over time. Within period I analyse, the 2008 downturn affects both variables similarly. It is possible, that in the longer period these variables are stationary, as the 2008 shock does not have a permanent impact on its level. Both unemployment rate and underemployment index come back to its pre-downturn level in 2017 q3.

Table 5 Summary Statistics

Variable	Number of observations	Mean	Standard Deviation	Minimum Value	Maximum Value
Underemployment index	71	6.22	2.01	3.87	9.79
Unemployment rate	71	5.94	1.28	4.26	8.39
Real Wage	71	457.34	37.19	359.13	504.92

According to Box and Tiao (1975) I should have at least 50, ideally more than 100 observations for time series analysis. Time series consists of 71 observations which is sufficient for analysis. Summary statistics of three times-series are reported in Table 5.

In order to test for stationarity, I initially run Augmented Dickey-Fuller unit root test and confirm findings with Phillips-Perron test. The advantage of Phillips-Perron test is that it controls for heteroskedasticity, which might be in error term in the augmented Dickey-Fuller test. Augmented Dickey-Fuller test has the null hypothesis that a variable contains a unit root. If null is rejected, I may assume that given time series is stationary (Hamilton, 1994).

Running Augmented Dickey-Fuller test requires specification of number of lags (p). If p is too small, Augmented Dickey-Fuller test might be biased. Too large p will result is loss of power of a test. I choose number of lags according to Schwarz Information Criteria. I also need to specify maximum number of lags before obtaining lag-order selection statistics. I set maximum lag number according to the formula suggested by Schwert (1989): p_max  =[12×(T/100)^0.25 ]. It gives maximum number of lags approximately 11 in case of 71 observations. I round this number up, as Monte-Carlo Experiments suggests that it is better to error on the side of including too many lags (Faculty Washington, 2018). I keep default confidence interval of 0.95.

Augmented Dickey-Fuller tests finds that the real wage is integrated of order two, unemployment rate and underemployment index are integrated of order one (Table 20 - Table 26 in appendices). I confirm these findings with Phillips-Perron tests. It is rather unusual to find macroeconomic variables, which need to be integrated twice. Mosconi, Rocco, and Paruolo (2017) finds variables such as: models of stock and flow; inventories; consumption; income to be integrated of order two. I would expect real wage to be trend stationary, or integrated of order one.

Conventional unit root tests I used are biased toward a false unit root null, when the data are trend stationary with a structural break (Perron, 1989). I need to take into consideration existing trends and structural breaks in my data. If not, I may fail to reject the null hypothesis that the time series contains a unit root (Perron, 1989). Perron suggests to incorporate a breakpoint into the regression model and apply tests similar to the standard Dickey-Fuller unit root test.

4.2.2 Real Wage Stationarity

I set up real wage structural break stationarity test (Perron, 1989) based on following description. Between 2000 q2 and 2008 q3 real wage has trending behaviour (Graph 5). As a consequence of the 2008 downturn it shifts to a non-trending behaviour (there is trend break and level shift). The 2008 q3 downturn causes a structural break in wages (I know the break date) and break occurs immediately.

I specify model (4) of trending data with intercept and trend break. Firstly, I perform unit root test on raw data (Level). I choose lag number based on Bayesian Information Criteria, set maximum lag to 11 (Schwert, 1989). I choose break type to be additive outlier (based on graphical analysis), which is a one-off effect of a break (Patterson, 2000).



I assume there is a structural break in the third quarter of 2008 (when Lehman Brothers collapses – peak point of the 2008 downturn) and denote it by Tb. DUt(Tb) is an intercept variable which takes value one once structural break Tb occurs. DTt(Tb) is a trend break variable which occurs for t greater, or equal Tb (Perron, 1989). I apply monotonic (natural logarithm) transformation to the real wage.

Table 6 Unit Root Test with a Breakpoint: Log Real Wage

Test Statistics	1%
Critical Value	5%
Critical Value	10%
Critical Value	Probability
-2.96	-4.88	-4.24	-3.96	0.10

In my model, the null hypothesis is that time series contains a unit root with change in the level and slope. If I reject it, I assume that the series is trend stationary with changes in the intercept and the slope. I can not reject the null hypothesis, that the model contains a unit root, as the p value of 0.10 is greater than 0.05 level (Table 6).

Table 7 Unit Root Test with a Breakpoint: Log Real Wage (First Difference)

Test Statistics	1%
Critical Value	5%
Critical Value	10%
Critical Value	Probability
-12.52	-4.87	-4.23	-3.95	.01

I take first difference of natural logarithm real wage. I assume again model (4) with trending data (the time series exhibits trend before 2008 q3 – Graph 7), constant and the intercept break DUt(Tb) – 2008 q3. I found real wage to be integrated of order one, with change in the intercept and the trend in 2008 q3. I reject the null hypothesis that the series contains a unit root, since the p-value is at a 1% level of significance (Table 7).

Graph 7 Real Wage Growth. British Pounds, q2 2000 to q4 2017, Deflated by CPI 100 = 2015
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture4.png"></a>
</figure>


4.2.3 Unemployment Rate and Underemployment Index Stationarity

Standard Phillips-Perron unit root test and Augmented Dickey-Fuller test find unemployment rate and underemployment index to be integrated of order 1. Both time series appear to have structural break as a consequence of the economic crisis. After the 2008 downturn, the time series exhibit negative trend (Graph 6). I perform unit root test with a break point and change in trend as specified in equation (4) assuming structural break to occur in the third quarter of 2008. I express both time series in natural logarithm.

Table 8 Unit Root Test with a Breakpoint: Log Unemployment Rate

Test Statistics	1%
Critical Value	5%
Critical Value	10%
Critical Value	Probability
-2.36	-4.88	-4.24	-3.96	.50




Table 9 Unit Root Test with a Breakpoint: Log Unemployment Rate (First Difference)

Test Statistics	1%
Critical Value	5%
Critical Value	10%
Critical Value	Probability
-4.86	-4.33	-3.75	-3.45	.01

Table 10 Unit Root Test with a Breakpoint: Log Underemployment Index

Test Statistics	1%
Critical Value	5%
Critical Value	10%
Critical Value	Probability
-2.02	-4.88	-4.24	-3.96	.50

Table 11 Unit Root Test with a Breakpoint: Log Underemployment Index (First Difference)

Test Statistics	1%
Critical Value	5%
Critical Value	10%
Critical Value	Probability
-3.91	-4.33	-3.75	-3.45	.05

Graph 8 Underemployment Index (blue) Growth and Unemployment Rate (red) Growth
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture5.png"></a>
</figure>


I cannot reject null hypothesis that unemployment rate and underemployment index contain a unit root, as specified in model (4), as p-values are 0.50 (Table 8 and Table 10).


Next, I take first difference of log underemployment index and log unemployment rate. I assume a model (7) with non-trending data, constant, and an intercept break DUt(Tb) – 2008 q3 (see Graph 8). P-values of 0.01 and 0.05 for log unemployment rate growth and log underemployment index growth respectively (Table 9 and Table 11) allow to reject the null hypothesis. I find both time series to be integrated of order one, with change in the intercept in the third quarter of 2008.


4.3 Obtaining Residuals

In order to perform analysis of the relationship between real wage growth and underemployment index/unemployment rate growth I need to obtain residuals of those variables, which account for changes in trends and intercepts. I base this approach on the Frisch-Waugh-Lovell theorem. Time series have either change in trend and intercept (real wage growth), or just intercept (unemployment rate growth and underemployment index growth) in the third quarter of 2008.

In order to obtain residuals of real wage (ε_(w,t)), I will regress first difference of log real wage on change in intercept DUt(Tb) and time trend DT_t (T_b ) – see equation (8).


Both underemployment index and unemployment rate are integrated of order one once I exclude intercept break. In order to obtain residuals (ε_(und,t) and ε_(u,t)) I regress underemployment index growth and unemployment rate growth on constant and intercept break 〖DU〗t (T_b ) – equations (9) and (10).

(9)	 〖∆ln(u〗t)=α_2+δ_21 〖DU〗t (T_b )+ε_(u,t)

Graph 9 Real Wage Growth Residuals (First Difference of Natural Logarithm). q2 2000 to q4 2017, deflated by CPI 100 = 2015
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture6.png"></a>
</figure>

Graph 10 Unemployment Rate Growth Residuals (First Difference of Natural Logarithm). q2 2000 to q4 2017
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture7.png"></a>
</figure>



Graph 11 Underemployment Index Growth Residuals (First Difference of Natural Logarithm). q2 2000 to q4 2017. Source: ONS
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture8.png"></a>
</figure>




4.4 Exploratory Time Series Regression

I follow methodology of analysing sensitivities published by Gregg and Fernandez-Salgado (2014). In particular, I focus on running two exploratory time series regressions at the economy-wide level.

(11)	ln⁡〖(W_t )=〗 α_1+δ_1 ln(U_(t-1))+λ_1 t+η_t

Equation (11) from their paper relates the log real wage in period t solely to the log unemployment rate in period (t - 1) and a trend. Unemployment rate is lagged one period “to reduce the potential for current prevailing economic conditions to be both driving unemployment and wage movements” (Gregg and Fernandez-Salgado, 2014).

I run regression based on above specification for both unemployment rate and underemployment index. I expect my results to be different to Gregg and Fernandez-Salgado, as I analyse quarterly data, shorter period, and work on first differences.

Natural logarithms of wages and underemployment index are not stationary within period I analyse. In order to analyse similar relationship to (11) I need to modify this model. I try to establish short-run relationship, where all data is stationary. I regress residuals of first difference of log real wage on residuals of first difference of lagged log underemployment index/unemployment rate – as defined in subsection 4.3. I suppress constant term and report robust standard errors (robust to heteroskedasticity – as I work with survey data).

(12)        ε_(w,t)=δ_2 ε_(und,t-1)+υ_1t

(13)        ε_(w,t)=δ_3 ε_(u,t-1)+υ_2t

Table 12 Exploratory Time Series Regression (12) and (13)

Regression of Real Wage Residuals on Underemployment Index/Unemployment Rate Residuals*
Underemployment Index (t-1)	-.0308
(.0262)
Unemployment Rate (t-1)	-.0178
(.0298)
Robust standard errors in parentheses.

Table 12 shows correlations between real wage growth and lagged underemployment index/unemployment rate growths. Gregg and Fernandez-Salgado (2014) find that there is a significant wage restraining impact of lagged log unemployment on median log real wages (equation 11: δ_1= -.184 for 2003-2012). I find that for period (q2 2000 to q4 2017) there is a negative relationship between lagged unemployment growth and real wage growth (-.0178). 1% increase in lagged unemployment growth decreases real wage growth (response variable) by -.0178 (approximately 1.8%). This finding is in line with theory (Phillips, 1978) and intuition. Higher lagged unemployment growth has negative impact on real wage growth, as employees have lower bargain power due to excess supply on the labour market.

More interestingly, coefficient of lagged underemployment index growth is -.0308 (approximately negative 3.1%). It means that growth in real wage is significantly more responsive to changes in underemployment index growth, than changes in unemployment rate growth. This result is in line with my expectations, as underemployment index includes more information about the slack on the labour market than unemployment rate. The coefficient of lagged underemployment index growth is almost twice as large as unemployment rate growth. This is promising indicator, as it shows that underemployment index might be more accurate predictor of wage movements.

(14)     ln⁡〖(W_t )=〗 α_2+β_2 ∆ln(U_t )+δ_4 ln(U_(t-1))+λ_2 t+υ_t

In equation (14), from Gregg and Fernandez-Salgado (2014) paper, beta coefficient allows for “short-run effects of changes in log unemployment to affect log real wages” (Gregg and Fernandez-Salgado, 2014).

I run regression (14) expressed in terms of residuals of real wage growth and unemployment rate/underemployment index growth accounting for changes in the intercept and linear trend (see equations (15) and (16)). I again supress constant term and report robust standard errors.

(15)    ε_(w,t)=β_21 ε_(und,t)+δ_21 ε_(und,t-1)+υ_3t

(16)    ε_(w,t)=β_22 ε_(u,t)  + δ_22 ε_(u,t-1)  + υ_4t

Both lagged underemployment index growth and lagged unemployment rate growth have negative impact on real wage growth (Table 13). Surprisingly, underemployment index growth in current period t has positive impact on real wage growth, while unemployment rate growth in current period has negative effect. It might be the case that employers do not have enough time to adjust wages to current labour market conditions, which are included in underemployment index, as those data are not published by the official statistical institute, as well as are not so easily accessed at the aggregate level. However, unemployment data are in line with Gregg and Fernandez-Salgado (2014) results. The current period changes are badly determined by model (15), which suggests that I “did not pick up effects of economic cycle on real wages other than by lagged measures of slack growth” (Gregg Fernandez-Salgado, 2014).

Table 13 Exploratory Time Series Regression (15) and (16)

Regression of Real Wage on Underemployment Index, or Unemployment Rate*
Underemployment Index (t)	.0165
(.0279)
Underemployment Index (t-1)	-.0380
(.0282)
Unemployment Rate (t)	-.0286
(.0418)
Unemployment Rate (t-1)	-.0018
(.0393)
Robust standard errors in parentheses.

Above results suggest, that it is possible that both unemployment rate growth and underemployment index growth negatively impacts real wage growth. When I lag both indicators of slack of the labour market, they explain wage movements in line with macroeconomic theory and my expectations. However, results are ambiguous for current period changes. In order to explore relationship in more detail, I will run the Granger causality tests.


4.5 VAR Models

Before I run Granger causality tests, I need to construct VAR models. VAR analysis is a regression analysis involving stationary variables, where the past values of the variables are allowed to affect each other (Hamilton, 1994). I construct two bivariate VAR models with p and k lags.

I base specification of VAR model on Gregg and Fernandez-Salgado (2014) paper. I build two VAR models (17) and (18). Model (17) includes real wage growth and underemployment index growth. Model (18) includes real wage growth and unemployment rate growth. Models traces the dynamics interaction of slack on labour market and real wage (Hamilton, 1994).

This VAR analysis is an atheoretical analysis of short-term relationship between underemployment index/unemployment rate and real wages (Hamilton, 1994). I do not use labour market theory to explicitly specify the structural relationship between variables. Instead, I just decide which variables I want to include in VAR model. I base analysis on assumption, that indicators of slack on the labour market and real wages move together over time and there is some autocorrelation between those macroeconomic variables (Hamilton, 1994).

VAR approach to model the relationship between time series variables has several drawbacks. It is sensitive to lag-selection – I discuss approach towards this below. VAR results also do not allow to differentiate between correlation and causality. In order to overcome this problem, I run Granger Causality Tests. VAR can be also highly dimensional, if researchers try to model relationship between many variables (Hamilton, 1994). It is not the case in this dissertation, as I include two variables in each VAR model.

The first step in building VAR model is selecting the optimal lag length according to selection-order criteria. Monte Carlo experiments suggests that it is better to error on the side of including too many lags. However, if number of lags is too large, then the power of a test will suffer (Hamilton, 1994). I choose number of lags based on pre-estimation VAR selection statistics. Firstly, I set maximum lag to according to the formula suggested by Schwert (1989) lagmax  =[12×(T/100)^0.25 ] which again states maximum number of lags to be 11, as I analyse T =  71 data points in each case. I keep default confidence interval of 0.95. In the beginning, I test the most parsimonious VAR model. Selection of lags is made according to four tests (final prediction error (FPE), Akaike’s information criterion (AIC), Schwarz’s Bayesian information criterion (SBIC), and the Hannan and Quinn information criterion (HQIC)). Selection is done under the null hypothesis is that all the coefficients on the p-th/k-th lags of the endogenous variables are zero.

Below, I briefly explain how I choose VAR models (17) and (18) - testing for stability and autocorrelation.

4.5.1 VAR Model of Real Wage and Underemployment Index

I want to choose lag order of model (17). I start with the most parsimonious model of lag 1 (selected by SBIC and HQIC). I set up VAR model, and test for stability and for serial correlation.

Although model passes test for stability (all eigenvalues lie inside the unit circle – Graph 12), I need to reject the null hypothesis that there is no autocorrelation (Table 14). Evidence of autocorrelation means, that model requires re-estimation.

Graph 12 Stability of VAR(1) Model (17)
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture9.png"></a>
</figure>





Table 14 Testing for Serial Correlation of VAR (1) Model (17)

Lag	Chi2	Df	Prob>Chi2
1	10.0790	4	0.03912
2	8.3559	4	0.07938

I re-estimate model with 2 lags (selected by AIC and FPE). I set up VAR model and test for stability and serial autocorrelation. Model pass test for stability (Graph 13) and there is no evidence of autocorrelation (Table 15) – P-values are greater than 0.05, thus I do not reject the null hypothesis of no autocorrelation at lag order. Vector autoregression model (17) with two lags is chosen.

Graph 13 Stability of VAR(2) Model (17)
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture10.png"></a>
</figure>


Table 15 Testing for Serial Correlation of VAR (2) Model (17)

Lag	Chi2	Df	Prob>Chi2
1	1.2131  	4	0.87593
2	4.4668	4	0.34650  


4.5.2 VAR Model of Real Wage and Unemployment Rate

Now, I am choosing lag order of model (18). I start with the most parsimonious model of lag 1 (selected by FPE, AIC, HQIC, SBIC). I set up VAR model and test for stability and for serial correlation. Model passes test for stability, as all eigenvalues lie inside the unit circle (Graph 14). I do not reject the null hypothesis that there is no autocorrelation, as p-values of autocorrelation tests are greater than 0.05 at each lag (Table 16). Vector autoregression model (18) with one lag is chosen.

Graph 14 Stability of VAR(1) Model
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/Picture11.png"></a>
</figure>


Table 16 Testing for Serial Correlation of VAR (1)

Lag	Chi2	Df	Prob>Chi2
1	5.1842	4	0.26892
2	5.4865	4	0.24092



4.6 Granger Causality Tests

Granger causality test is a way to investigate if one variable Granger-cause another variable. Given two stationary variables x, y and correct VAR model, this test reveals if variable x helps to predict variable y, or if variable y helps to predict variable x, or both (Patterson, 2000). So far, I ran exploratory regressions, which attempts to establish type of relationship between the topic variables. Granger causality test allows establishing both if there is any relationship between variables and direction of this relationship. Granger causality test run under the null hypothesis that e.g. variable x does not granger-cause variable y. If null hypothesis is rejected, then I conclude that causality of x to y exist. This method is a “probabilistic account of causality and it finds patterns in correlation” (Patterson, 2000). I define null hypothesis of Granger causality tests based on VAR models: (17) with p = 2 lags and (18) with k = 1 lag.

Null hypothesis for VAR (2) model (17):

(19)	H0: δ_112= δ_212=0
(20)	H0: δ_121=δ_221=0

Null hypothesis for VAR (1) model (18):

(21)	H0: ϕ_112=0
(22)	H0: ϕ_121=0

Table 17 Granger Causality Test, VAR(2) Underemployment Growth – Real Wage Growth

Equation	Excluded	Chi2	df	Prob>Chi2
Real Wage*	Underemployment Index*	1.1064	2	.575
Underemployment Index*	Real Wage*	3.2856	2	.193

Results of the Granger causality test based on VAR(2) model (17) show, that I cannot reject the null hypothesis that underemployment index growth does not Granger-cause real wage growth (Table 17). Similarly, I cannot reject the null hypothesis that real wage growth does not Granger-cause underemployment index growth. It might be worth to note, that p value of Granger-causation from real wage growth to underemployment index growth is more significant than the opposite direction, but the significance in not at any reasonable level.

Table 18 Granger Causality Test, VAR(1) Unemployment Growth –Real Wage Growth
<figure>
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/images/DIS/1.png"></a>
</figure>

Equation	Excluded	Chi2	df	Prob>Chi2
Real Wage*	Unemployment Rate*	.6298	1	0.427
Unemployment Rate*	Real Wage*	.7123	1	0.427
growth

Next, I run Granger causality test on VAR(1) model (18). Result (Table 18) show that I cannot reject the null hypothesis, that real wage growth does not Granger-cause unemployment rate growth. I also cannot reject the null hypothesis that unemployment rate growth does not Granger-cause real wage growth.

Although the results of exploratory time series regression discussed in subsection 4.4 are promising, Granger causality tests do not indicate any relationship between underemployment index growth and real wage growth. Tests also do not detect any relationship between unemployment rate growth and real wage growth. Latter findings are in line with results of Gali (2011), who also performed Granger causality tests between quarterly unemployment rate growth and real wage growth (1964 q1 – 2009 q1, US data), but did not find any significant causation.

Some of previous researchers find that movements in unemployment rate affect real wage in selected periods (see section 2.1 and Table 1). I cannot clearly state that there is a relationship between real wage growth and underemployment index growth between 2000 q2 – 2017 q4, as I do not have enough evidence to support this statement. Exploratory regression results obtained in subsection 4.4 are not sufficient, as Gregg and Fernandez-Salgado use them just as a possible indicator of the relationship. Researchers are not trying to obtain model, which fully explains how real wage movements are determined. They just want indication of direction in which real wage moves, given change in unemployment rate. I am also certain that models proposed by Gregg and Fernandez-Salgado (2014) suffer from omitted variable bias, as there are many various factors affecting real wage growth. I try to override this problem by establishing Granger causation. Establishing Granger causation could be the sufficient evidence. Although VAR models may still suffer from omitted variable bias, Granger causation could simply indicate if e.g. underemployment index growth changes help to predict real wage growth changes. Then, I would be able to establish the relationship between topic variables at least in a statistical sense. As shown above, it is not the case within the period I analyse.
 


``` STATA
//Setting time in time series in stata

generate time = q(2000q2) + _n -1
format t %tq

tsset time

rename var1 und
label variable und "Underemployment Index"

rename var3 u
label variable u "Unemployment Rate"

label variable w "Real Wage"

rename var4 h
label variable h "Hours"

var r_dlw r_dlu, lags(1)
var r_dlw r_dlu, lags(1)
varstable, graph graphregion(color(white)) bgcolor(white)
varlmar


tsline  und u, /// plot all predictions on one graph (5 seris)
title("Underemployment Index and Unemployment Rate") ///
subtitle("q2 2000 to q4 2017") xtitle("") ytitle("%") graphregion(color(white)) bgcolor(white)
graph export U-UND.png //safe graphs to your drive in .png


tsline  u var18 var19 if time>time[17], /// plot all predictions on one graph (5 seris)
title("Underemployment Rate and Unemployment Rate") ///
subtitle("q1 2002 to q4 2017") xtitle("") ytitle("%") graphregion(color(white)) bgcolor(white)

tsline  dlw, /// plot all predictions on one graph (5 seris)
title("Real Wage growth deflated by CPI 2015=100") ///
subtitle("q2 2000 to q4 2017") xtitle("") ytitle("British Pounds") graphregion(color(white)) bgcolor(white)

tsline  r_dlund, /// plot all predictions on one graph (5 seris)
title("Underemployment Index Growth Residuals") ///
subtitle("q2 2000 to q4 2017") xtitle("")  graphregion(color(white)) bgcolor(white)


twoway bar h time, yscale(range(40 80)) title("Sum of Underemployed and Overemployed hours") ///
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Millions") graphregion(color(white)) bgcolor(white) ///
ylabel(40 50 60 70 80)

generate neg_1 = -


twoway (bar neg_1 t, barwidth(0.5) ) (bar increases t, barwidth(0.5))

twoway bar reductions neg_1 t, barwidth(0.5)

graph bar reductions increases t, stack

yscale(range(40 80)) title("Sum of Underemployed and Overemployed hours") ///
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Millions") graphregion(color(white)) bgcolor(white) ///
ylabel(40 50 60 70 80)

yscale(range(40 80)) title("Sum of Underemployed and Overemployed hours") ///
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Millions") graphregion(color(white)) bgcolor(white) ///
ylabel(40 50 60 70 80)

twoway bar h time || rbar reductions h time

twoway bar reductions time || rbar h reductions time

twoway bar reductions time, barwidth(0.5)  || rbar h reductions time , yscale(range(0 80)) title("Sum of Underemployed and Overemployed Hours") ///
subtitle("q2 2000 to q4 2017") xtitle("") ytitle("Millions") graphregion(color(white)) bgcolor(white) ///
ylabel(0 10 20 30 40 50 60 70 80) barwidth(0.5)

tsline dlnw,  graphregion(color(white)) bgcolor(white) title("Real Wage Growth")
subtitle("q2 2000 to q2 2017, CPI 2015=100") xtitle("") ytitle("") graphregion(color(white)) bgcolor(white) //


tsline dlnund,  ytitle("Growth rate") xtitle("") graphregion(color(white)) bgcolor(white) title("Underemployment Index Growth")
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Growth rate") graphregion(color(white)) bgcolor(white) //

tsline dlnu,  ytitle("Growth rate") xtitle("") graphregion(color(white)) bgcolor(white) title("Unemployment Rate Growth")
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Growth rate") graphregion(color(white)) bgcolor(white) //

ac ddlnw, ytitle("Autocorrelation") xtitle("") graphregion(color(white)) bgcolor(white)

ac ddlnund, ytitle("Autocorrelation") xtitle("") graphregion(color(white)) bgcolor(white)

ac dlnu, ytitle("Autocorrelation") xtitle("") graphregion(color(white)) bgcolor(white)

tsline ddlnw,  ytitle("Change in Growth Rate") xtitle("") graphregion(color(white)) bgcolor(white) title("Real Wage Change in Growth")
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Growth rate") graphregion(color(white)) bgcolor(white) //

tsline ddlnund,  ytitle("Change in Growth Rate") xtitle("") graphregion(color(white)) bgcolor(white) title("Underemployment Index Change in Growth")
subtitle("q2 2000 to q2 2017") xtitle("") ytitle("Growth rate") graphregion(color(white)) bgcolor(white) //


```
