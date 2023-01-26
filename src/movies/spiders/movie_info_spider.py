import re

import pandas
import scrapy


class MovieInfoSpider(scrapy.Spider):
    """
    Spider responsible for storing movies to be further scraped
    """
    name = "MOVIE_INFO_SPIDER"
    download_delay = 2

    def start_requests(self):
        url_start = 'https://en.wikipedia.org'
        url_list = pandas.read_csv('films_part_5.csv')
        print(url_list)
        for row in url_list.iterrows():
            print(row[1]['year'])
            yield scrapy.Request(url_start + row[1]['link'], callback=self.parse_wiki,
                                 cb_kwargs=dict(year=row[1]['year']))

    def parse_wiki(self, response, year):
        print(year)
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

        rotten_tomatoes_url_title = re.sub(' ', '_', movie_title)
        rotten_tomatoes_url_title = re.sub('\W', '', rotten_tomatoes_url_title)
        rotten_tomatoes_praefix = 'https://www.rottentomatoes.com/m/'

        rotten_tomatoes_url = rotten_tomatoes_praefix + rotten_tomatoes_url_title
        yield scrapy.Request(rotten_tomatoes_url + '_' + str(year), callback=self.parse_rotten_tomatoes,
                             cb_kwargs={'wikidict': wikidict, 'retry_url': rotten_tomatoes_url})

    def parse_rotten_tomatoes(self, response, wikidict, retry_url):
        if response.status == 404:
            if retry_url is not None:
                yield scrapy.Request(retry_url, callback=self.parse_rotten_tomatoes,
                                     cb_kwargs={'wikidict': wikidict, 'retry_url': None})
            else:
                print('No rotten tomatoe for ', wikidict['title'])
        else:
            # critics:
            critics_count = response.selector.xpath('//a[@slot="critics-count"]/text()').get()
            critics_score = response.selector.xpath('//score-board[@class="scoreboard"]/@tomatometerscore').get()

            # audience:
            audience_count = response.selector.xpath('//a[@slot="audience-count"]/text()').get()
            audience_score = response.selector.xpath('//score-board[@class="scoreboard"]/@audiencescore').get()

            # get suplieres:
            suppliers = response.selector.xpath('//where-to-watch-meta/@affiliate').getall()
            suppliers_list = []
            for supplier in suppliers:
                type = response.selector.xpath(
                    f'//where-to-watch-meta[@affiliate="{supplier}"]/span[@slot="license"]/text()').get()
                suppliers_list.append((supplier, type))

            info_str = response.selector.xpath(
                '//score-board[@class="scoreboard"]/p[@class="scoreboard__info"]/text()').get()
            info_list = info_str.split(',')
            rottentomatoes_year = info_list[0]
            rottentomatoes_genre = info_list[1]
            rottentomatoes_length = info_list[2]

            if int(rottentomatoes_year) != wikidict['year']:
                print('Falsche Seite:' + response.url)
            movie_info_dict = wikidict
            movie_info_dict['critics_count'] = critics_count
            movie_info_dict['critics_score'] = critics_score
            movie_info_dict['audience_count'] = audience_count
            movie_info_dict['audience_score'] = audience_score
            movie_info_dict['suppliers_list'] = suppliers_list
            movie_info_dict['rottentomatoes_year'] = rottentomatoes_year
            movie_info_dict['rottentomatoes_genre'] = rottentomatoes_genre
            movie_info_dict['rottentomatoes_length'] = rottentomatoes_length

            yield movie_info_dict

    def clean_spaces(self, s):
        s = re.sub('[ \n]*$', '', s)
        s = re.sub('^[ \n]*', '', s)
        return s
