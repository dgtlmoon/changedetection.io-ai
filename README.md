# AI / Machine Learning Helper for Changedetection.io

<img src="docs/AI-MachineLearning-scraper.jpg">

Licence: Used under the same dual-licencing as changedetection ( [Apache2.0](COMMERCIAL_LICENCE.md), [Commercial](COMMERCIAL_LICENCE.md) )

So far this project contains one AI/ML helper, but more to come.

# Automatically identify page elements containing price information

**Problem solved**: A product page does not contain any semantic data (LD-JSON product data etc) and you want the price data for the product.
Fortunately the price "text" is generally always of nice text size, color, font weight, position on the screen, contains 
digits etc.

Using those attributes (text size, weight etc) we can train a very simple ML model using pytorch/keras that is 99% accurate.

Simply use this application as the `PRICE_SCRAPER_ML_ENDPOINT` from your [changedetection.io](https://github.com/dgtlmoon/changedetection.io) endpoint. 
(@todo link to tutorial)

Changedetection.io will query this service by sending its specially scraped data (a long list of information about each 
DOM element such as font colour, weight, text size etc) and return what it thinks is the best match.


