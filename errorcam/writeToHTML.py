'''
@author: Arijit Ray
Support file for writing stuff to HTML files in python so that people can visualize stuff. 
Beware: simple and hacky : here there be experiments. Output might not look like the prettiest webpage in the world. 
10/10/16
'''

import os
import datetime
from random import shuffle
import pdb

class HTMLPage:
    def __init__(self, filename, title, append=False):
        self.fileName = filename
        if append == True:
            with open(self.fileName, "a") as f:
                print("<p> appending to file at :" + str(datetime.datetime.now()) + " </p>", file=f)
        else:
            with open(self.fileName, "w") as f:
                s='<meta charset="utf-8">\
                    <meta name="viewport" content="width=device-width, initial-scale=1">\
                    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">\
                    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>\
                    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>'
                print(s, file=f)
                print("<title>", file=f)
                print(str(title), file=f)
                print("</title>", file=f)
                print("<head>", file=f)
                print("<h2> " + str(title) + "</h2>", file=f)
                print("<p> Run Date and Time: " + str(datetime.datetime.now()) + " </p>", file=f)
                print("</head>", file=f)
                print("<body><div class='container-fluid'>", file=f)
        self.RowStarted = False
        self.TableStarted=False
        self.ColStarted=False
        self.colHeader = False

    def writeImage(self, imageFile, width=200, height=200):
        with open(self.fileName, 'a') as f:
            print("<img src=\"" + imageFile + "\" width=" + str(width) + " height=" + str(height) + ">", file=f)

    def writeTextList(self, questionPool):
        with open(self.fileName, 'a') as f:
            for question in questionPool:
                print(" &nbsp; " + question + " &nbsp; ", file=f)

    def writeText(self, question):
        with open(self.fileName, 'a') as f:
            print(" " + str(question) + " ", file=f)

    def breakLine(self):
        with open(self.fileName, 'a') as f:
            print("<br/>", file=f)

    def horizontal_line(self):
        with open(self.fileName, 'a') as f:
            print("<hr/>", file=f)

    def startTable(self):
        with open(self.fileName, 'a') as f:
            if self.TableStarted:
                self.endTable()
            self.TableStarted = True
            print('<table width="100%" cellpadding="3px">', file=f)

    def endTable(self):
        self.TableStarted = False
        with open(self.fileName, 'a') as f:
            if self.ColStarted:
                if self.colHeader:
                    print("</th>", file=f)
                else:
                    print("</td>", file=f)
                self.ColStarted=False

            if self.RowStarted:
                print("</tr>", file=f)
                self.RowStarted = False
            print("</table>", file=f)

    def startRow(self):
        with open(self.fileName, 'a') as f:
            if self.RowStarted:
                print("</tr>", file=f)
            self.RowStarted = True
            print("<tr>", file=f)

    def endRow(self):
        self.RowStarted = False
        with open(self.fileName, 'a') as f:
            print("</tr>", file=f)

    def startCol(self, header=False):

        with open(self.fileName, 'a') as f:
            if self.ColStarted:
                if self.colHeader:
                    print("</th>", file=f)
                else:
                    print("</td>", file=f)
            self.colHeader = header
            self.ColStarted=True
            if header == False:
                print("<td>", file=f)
            else:
                print("<th>", file=f)

    def endCol(self):
        self.ColStarted = False
        with open(self.fileName, 'a') as f:
            if self.colHeader == False:
                print("</td>", file=f)
            else:
                print("</th>", file=f)

    def closeHTMLFile(self):
        with open(self.fileName, 'a') as f:
            print("<hr />", file=f)
            print("</div></body>", file=f)
            print("</html>", file=f)

    def make_bar_chart(self, rating_array, rating_labels, max_value):
        #assumes a list of values
        s=""
        for entry, label in zip(rating_array, rating_labels): 
            s+='<div class="progress">\n'
            s+='\t<div class="progress-bar" role="progressbar" aria-valuenow="'+str(entry)+'" aria-valuemin="0" aria-valuemax="'+str(max_value)+'" style="width:'+str(entry)+'%">\n'
            s+=label + " " +str(entry)
            s+='</div></div>\n'

        with open(self.fileName, 'a') as f:
            print(s, file=f)

    
    ### AMT Support

    def make_radiobutton_question(self, name, options, on_click_functions=None):

        return s

    def make_multichoice_question(self, name, options, on_click_functions=None):

        return s

    def make_button(self, action, name, on_click_function=None):

        return s