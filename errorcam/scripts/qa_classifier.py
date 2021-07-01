from collections import defaultdict
import json
import numpy as np
import csv


def text2int(textnum, numwords={}):
        #w2v = self.w2v
        '''
        Code adapted from :
        https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers answer by recursive username. 
        '''

        if not numwords:
          units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
          ]
    
          tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
          scales = ["hundred", "thousand", "million", "billion", "trillion"]
    
          numwords["and"] = (1, 0)
          for idx, word in enumerate(units):    numwords[word] = (1, idx)
          for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
          for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
    
        current = result = 0
        for word in textnum.split():
            if word not in numwords:
              raise Exception("Illegal word: " + word)
    
            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
    
        return result + current


class qa_classifier:

    def __init__(self):

        action_vocab_file = "scripts/chosenActions.csv"
        self.all_actions=[]
        with open(action_vocab_file) as f:
            reader = csv.reader(f)
            chosenActions = list(reader)
            chosenActions=chosenActions[0]
            chosenActions.append("taking")
            chosenActions.append("jumping")
        self.all_actions = chosenActions

        object_classes_file = "scripts/topObjVQA_VG_Intersect_List.csv"
        self.all_objects =[]
        with open(object_classes_file) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                self.all_objects.append(row[0].strip().lower())

        self.time_classes = ["dawn", "morning", "noon", "afternoon", "dusk", "evening", "night", "daytime", "nighttime", "day", "midnight", "summer", "autumn", "fall", "spring", "winter", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

        self.color_classes = ["red", "blue", "orange", "pink", "white", "black", "yellow", "green", "purple", "brown", "gray"]

        self.STOP_WORDS = ["the", "is", "and", "this", "image", "photo", "picture", "in", "are", "on", "that", "photograph", "then", "or", "does"]

        self.weather_words = ["sunny", "rainy"]

    
    def classify_qa(self, question, answer):

        #should return the type: one of count, action, color, object, weather, time

        word = answer.lower().split(" ")[0].split("'")[0].split(",")[0].split(".")[0].split("?")[0]
        if word in self.color_classes:
            return "color"
        
        if word in self.all_objects:
            return "object"

        if word in self.all_actions:
            return "action"
        
        if word in self.time_classes:
            return "time"

        try:
            num = float(word)
            return "count"
        except:
            try:
                num = float(text2int(word))
                return "count"
            except:
                _=1

        qa_words = question.strip().lower().split("?")[0].split(".")[0].split(" ")
        qa_key_list = []
        for w in qa_words:
            w = w.split("'")[0]
            if w not in self.STOP_WORDS:
                if w not in qa_key_list:
                    if w in ["man", "woman", "person", "child", "boy", "girl"]:
                        w="person"
                    qa_key_list.append(w)

        qa_key = " ".join(qa_key_list[:2])
        qtype_word = qa_key_list[2] if len(qa_key_list)>2 else "None"

        if qa_key == "how many":
            return "count"

        if qa_key=="what color":
            return "color"

        if qa_key=="what person" and qtype_word=="doing":
            return "action"

        if qa_key=="how weather" or word in self.weather_words:
            return "weather"


        return None