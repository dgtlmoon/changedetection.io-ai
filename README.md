# AI / Machine Learning Helper for Changedetection.io

<img src="docs/AI-MachineLearning-scraper.jpg" alt="Ai Machine Learning scraper for changedetection.io product prices" title="Ai Machine Learning scraper for changedetection.io product prices" />

Licence: Used under the same dual-licencing as changedetection ( [Apache2.0](COMMERCIAL_LICENCE.md), [Commercial](COMMERCIAL_LICENCE.md) )

So far this project contains one AI/ML helper, but more to come.

# Automatically identify page elements containing price information

Available for **linux/amd64**, more to come!

For use with changedetection.io **v0.46.04** and newer!


**Problem solved**: A product page does not contain any semantic data (LD-JSON product data etc) and you want the price data for the product.
Fortunately the price "text" is generally always of nice text size, color, font weight, position on the screen, contains 
digits etc.

Using those attributes (text size, weight etc) we can train a very simple ML model using pytorch/keras that is 99% accurate.

Simply use this application as the `PRICE_SCRAPER_ML_ENDPOINT` from your [changedetection.io](https://github.com/dgtlmoon/changedetection.io) endpoint. 
(@todo link to tutorial)

Changedetection.io will query this service by sending its specially scraped data (a long list of information about each 
DOM element such as font colour, weight, text size etc) and return what it thinks is the best match.

## Re-train the model for your exact needs

Although the model ships with quite a nice default, here's some notes for training with your own dataset.

- All selector rules (CSS/xPath) should point to the exact DOM element that contains the price value, it's important that it can find information such as `font-height` length of text etc.
- For web-pages that have the "cents" in a separate DOM element (Example: `<div>$10<span>.11</span></div>`, always point it to the main "dollars" selector which has the best font-size and other information
- Point `train_model.py` at the path to your datasource for your changedetection.io installation (where the `url-watches.json` file lives)
- Always only have "watches" setup with working selectors that only point to the price information
- Try to train on web-pages that have the price information in different places, ie - left, centre, low down below, right side etc

```
./train_model.py -e 30 -d /path/to/changedetection.io-data
```



### Nerdy stuff

**What is this?** It's just a _binary classification_ model for iterating over all of the scraped DOM elements and
returns the best match.

**Is it fast?** Oh yeah! According to Apache Benchmark

```bash
ab -p elements/set-12.json -T application/json -c 5 -n 100 http://127.0.0.1:5005/price-element

Concurrency Level:      5
Time taken for tests:   5.657 seconds
Complete requests:      100
Failed requests:        0
Requests per second:    17.68 [#/sec] (mean)
Time per request:       282.829 [ms] (mean)
Time per request:       56.566 [ms] (mean, across all concurrent requests)

```

**56ms per request!** that's fast! without GPU on a regular `Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz`

When tested with `curl`, its 0.06 seconds including the time to run the curl app and execute it.

```bash
$ time curl -s -X POST http://127.0.0.1:5005/price-element --data-binary @elements/set-12.json -H "Content-Type: application/json"|python -mjson.tool
{
    "bin": 1,
    "idx": 273,
    "score": 0.9991648197174072
}

real    0m0.062s
user    0m0.015s
sys     0m0.004s

```

`idx` = `273` means that it recommends using element #273 in the scraped list of elements.


Have fun!
