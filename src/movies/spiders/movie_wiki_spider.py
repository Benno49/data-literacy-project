import re

import pandas
import scrapy


class MovieInfoSpider(scrapy.Spider):
    """
    Spider responsible for storing movies to be further scraped
    """
    name = "MOVIE_WIKI_SPIDER"
    download_delay = 1

    def start_requests(self):
        url_start = 'https://en.wikipedia.org'
        url_list = pandas.read_csv('films_part_3.csv')
        print(url_list)
        for row in url_list.iterrows():
            print(row[1]['year'])
            yield scrapy.Request(url_start + row[1]['link'], callback=self.parse_wiki,
                                 cb_kwargs=dict(year=row[1]['year']))

    def parse_wiki(self, response, year):
        movie_title = response.selector.xpath('//h1[contains(@id,"firstHeading")]/i/text()').get()
        movie_title = re.sub('\(.*\)$', '', movie_title)
        movie_title = self.clean_spaces(movie_title)

        infobox = response.selector.xpath('//tr[th[@class="infobox-label"]]')
        infobox_dict = dict()
        for line in infobox:
            # für normale th:
            category_list = line.xpath('th/text()').getall()
            # für multilines:
            category_list += line.xpath('th/div[not(contains(@class,"plain-list"))]/text()').getall()
            category = ' '.join(category_list)

            # für normale th
            data_list = line.xpath('td/text()').getall()
            data_list += line.xpath('td/a/text()').getall()
            # für plain lists
            data_list += line.xpath('td/div[contains(@class,"plainlist")]/ul/li/text()').getall()
            data_list += line.xpath('td/div[contains(@class,"plainlist")]/ul/li/a/text()').getall()
            # für multilines
            multiline_list = line.xpath('td/div[not(contains(@class,"plain-list"))]/text()').getall()
            data_list.append(' '.join(multiline_list))
            multiline_ref_list = line.xpath('td/div[not(contains(@class,"plain-list"))]/a/text()').getall()
            data_list.append(' '.join(multiline_ref_list))
            # für produktion companies lists mit style div
            data_list += line.xpath(
                'td/div[not(contains(@class,"plain-list"))]/div[contains(@class,"plainlist")]/ul/li/text()').getall()
            data_list += line.xpath(
                'td/div[not(contains(@class,"plain-list"))]/div[contains(@class,"plainlist")]/ul/li/a/text()').getall()

            infobox_dict[category] = data_list

        wikidict = dict()
        wikidict['title'] = movie_title
        wikidict['year'] = year
        wikidict['infobox'] = infobox_dict

        yield wikidict

    def clean_spaces(self, s):
        s = re.sub('[ \n]*$', '', s)
        s = re.sub('^[ \n]*', '', s)
        return s
