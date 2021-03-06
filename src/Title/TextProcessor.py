
import re

class TextProcessor:

    def __init__(self):
        """ Baseline text processor"""

    def processTitle(self,titles):
        """ Input: List of titles [title,title,title,....]"""
        """ Output: List of titles """
        pro_titles = []
        char_reg = "!@#$%^*_+{}:<>?,.;\"="

        for i in range(len(titles)):

            title = titles[i]
            new_title = title
            new_title = new_title.lower()
            new_title = re.sub('\\([^>]+\\)', '', new_title)
            new_title = re.sub('\[[^>]+\]', '', new_title)
            new_title = re.sub('\u200b', '', new_title)
            new_title = new_title.replace(' u '," you ")
            new_title = new_title.replace('u,', "you")

            for ch in char_reg:
                new_title = new_title.replace(ch, "")

            head, sep, tail = new_title.partition('/')
            head, sep, tail = head.partition('-')
            head, sep, tail = head.partition('|')
            new_title = head
            new_title = new_title.rstrip()
            new_title = new_title.lstrip()

            pro_titles.append(new_title)

            # print(title," = ",new_title)
        return pro_titles