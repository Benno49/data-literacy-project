import scrapy
from scrapy.linkextractors import LinkExtractor


class MovieListSpider(scrapy.Spider):
    """
    Spider responsible for storing movies to be further scraped
    """
    name = "MOVIE_LIST_SPIDER"
    year_list_extractor = LinkExtractor(allow=[r"/wiki/List_of_American_films_of_\d+"])
    download_delay = 5

    def start_requests(self):
        url_start = 'https://en.wikipedia.org/wiki/'
        url_end = '_in_film'
        for year in range(1950, 2023):
            url = url_start + str(year) + url_end
            yield scrapy.Request(url=url, callback=self.parse_year, cb_kwargs=dict(year=year))

    def parse_year(self, response, year):
        # TODO extract only valid links
        link_list = response.selector.xpath(
            f'//div[@class="mw-parser-output"]/ul[preceding::h2[./span[@id="{year}_films"]]]/li/a[contains(text(), "List of")]/@href').getall()
        if year >= 2006 and year <= 2009:
            link_list = response.selector.xpath('//ul[following-sibling::h2/span[contains(@id,"Births")]]/li/a/@href').getall()
        if len(link_list) > 0:  # true for the later years
            link_prefix = 'https://en.wikipedia.org'
            for link in link_list:
                yield scrapy.Request(link_prefix + link, callback=self.parse_film_list, cb_kwargs=dict(year=year))
        else:
            print('else')
            for pair in self.parse_film_list(response, year):
                yield pair
            print('?')

    def parse_film_list(self, response, year):
        print('entered')
        # look for "Notable films released in" section and get links
        link_list = response.selector.xpath(
            f'//ul[preceding::h2[./span[@id="Notable_films_released_in_{year}"]] and following-sibling::h2[./span['
            '@id="Deaths"]]]/li/i/a/@href').getall()
        print('1')
        # TODO if not prev. look for https://en.wikipedia.org/wiki/List_of_Tamil_films_of_2005
        if len(link_list) == 0:
            link_list = response.selector.xpath(
                '//table[preceding-sibling::h2[./span[contains(@id,"List_of_")]]]/tbody/tr[preceding-sibling::tr/th['
                'contains(text(),"Opening")]]/td/i/a/@href').getall()
            print('2')
        # TODO if not prev. look for tabel with Title in header th column and link (new format)
        if len(link_list) == 0:
            link_list = response.selector.xpath(
                '//table/tbody/tr[preceding-sibling::tr/th[contains(text(),"Opening")]]/td/i/a/@href').getall()
            print('3')
        # TODO if not prev. look for https: // en.wikipedia.org / wiki / List_of_French_films_of_2005

        if len(link_list) == 0:
            link_list = response.selector.xpath(
                '//table/tbody[tr/th[contains(text(),"Title")]]/tr/td/i/a/@href').getall()
            print('4')

        # if not prev. look for Tamil films shema

        if len(link_list) == 0:
            link_list = response.selector.xpath(
                '//table/tbody[./tr/th[contains(text(),"Opening")]]/tr/td/i/a/@href').getall()
            print('5')

        # TODO if not prev. print url for checking what went wrong
        if len(link_list) == 0:
            print("Fail:", response.url)
            print('5')
        print(link_list)
        for link in link_list:
            yield {"link": link, "year": year}

# TODO check resulting list for duplicates
