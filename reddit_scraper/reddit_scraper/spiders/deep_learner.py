import scrapy


class DeepLearnerSpider(scrapy.Spider):
    name = 'deep_learner'
    allowed_domains = ['www.reddit.com']
    start_urls = ['http://reddit.com/r/deeplearning/']

    custom_settings = {
        'DEPTH_LIMIT': 20
    }

    def start_requests(self):
        self.index = 0
        for url in self.start_urls:
            # We make a request to each url and call the parse function on the http response.
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):
        # filename = "deep_learning_response" + str(self.index) + ".txt"
        # with open(filename, 'wb') as f:
        # #     # All we'll do is save the whole response as a huge text file.
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)
        post = response.xpath('//*[@class="_1qeIAgB0cPwnLhDF9XSiJM"]/text()').extract()
        yield {"Post": set(post)}

        next_page = response.xpath('//link[@rel="next"]/@href').extract_first()
        if next_page:
            yield response.follow(next_page, self.parse)
