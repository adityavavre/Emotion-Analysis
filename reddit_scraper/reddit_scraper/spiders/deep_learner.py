import scrapy


class DeepLearnerSpider(scrapy.Spider):
    name = 'deep_learner'
    allowed_domains = ['https://www.reddit.com/r/deeplearning/']
    start_urls = ['http://https://www.reddit.com/r/deeplearning//']

    def start_requests(self):
        self.index = 0
        for url in self.start_urls:
            # We make a request to each url and call the parse function on the http response.
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):
        filename = "kitten_response" + str(self.index)
        with open(filename, 'wb') as f:
            # All we'll do is save the whole response as a huge text file.
            f.write(response.body)
        self.log('Saved file %s' % filename)
