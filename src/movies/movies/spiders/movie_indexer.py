import scrapy
from scrapy.linkextractors import LinkExtractor
# from globals import MOVIE_LIST_SPIDER
from movies.URLs import URLS


class WikiMovieIndexSpider(scrapy.Spider):
    """
    Spider responsible for storing movies to be further scraped
    """
   # name = MOVIE_LIST_SPIDER
    year_list_extractor = LinkExtractor(allow=[r"/wiki/List_of_American_films_of_\d+"])

    def start_requests(self):
        urls = URLS
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # TODO extract only valid links
        for link in self.year_list_extractor.extract_links(response):
            yield scrapy.Request(link.url, callback=self.parse)
