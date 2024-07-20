import sys
from pathlib import Path
import oyaml as yaml
from prettytable import PrettyTable
from loremipsum import get_sentences
import argparse

"""This program is helper tool for folks who need to document YAML file. It will read YAML and generate tables out of 
each section of YAML - (as of now) it add two additional columns "itItRequired" and "description" 

It can generated (text) table or HTML table

Example :

Input YAML :
~~~~~~~~~~
metadata:
   name: nginx-deployment
    labels:
      app: nginx

Output Table :
~~~~~~~~~~~~
metadata:
+--------+------------------+----------+-----------------------------------------------------------+
| Field  | Example Value    | Required | Description                                               |
+--------+------------------+----------+-----------------------------------------------------------+
| name   | nginx-deployment |    --    | Lorem ipsum dolor sit amet, consecteteur adipiscing elit. |
| labels | --               |    --    | Lorem ipsum.                                              |
|   app  | nginx            |    --    | Lorem ipsum.                                              |
+--------+------------------+----------+-----------------------------------------------------------+	    

"""

parser = argparse.ArgumentParser(description='YAML file to (HTML) table converter',
                                 epilog='text table will be printed as STDOUT - html table will be save in html file ')
parser.add_argument('--inputFile', dest='inputfile', required=True, help="input yaml file to process")
parser.add_argument('--out', dest='format', choices=['txt', 'html', 'text'], help="convert yaml to text table or html "
                                                                                  "table")
args = parser.parse_args()

outputFmt = args.format
INPUT_YAML = args.inputfile

if outputFmt == 'text' or outputFmt == 'txt':
    PRINT_HTML = False
else:
    PRINT_HTML = True

in_file = Path(INPUT_YAML)
if not in_file.is_file():
    sys.exit("Input file [" + INPUT_YAML + "] does not exists")

SPACE_CHAR = '~'
OUTPUT_HTMl = INPUT_YAML.replace("yaml", "doc.html")

CSS_TEXT = """
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
          <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
          <style>
        body
        {
            padding-left: 20px;
        }
        th:nth-child(1) {
          width: 200px;
          }

        /* the second */
        th:nth-child(2) {
          width: 200px;
        }

        /* the third */
        th:nth-child(3) {
          width: 100px;
         }
         /* the third */
         th:nth-child(4) {
          width: 420px;
         }

         pre {
            white-space: -moz-pre-wrap; /* Mozilla, supported since 1999 */
            white-space: -pre-wrap; /* Opera */
            white-space: -o-pre-wrap; /* Opera */
            white-space: pre-wrap; /* CSS3 - Text module (Candidate Recommendation) http://www.w3.org/TR/css3-text/#white-space */
            word-wrap: break-word; /* IE 5.5+ */
            width: 725px
         }
          </style>
        </head>
        """


def listToString(inList):
    """ Convert list to String """
    ret = ""
    for line in inList:
        ret = ret + line
    return ret


def printDic(inDictionary, inPTable, indent):
    """
        Iterate over Dictionary
           If needed call same function again (recursively) until we key : value dictionary
           Add key , value , isItRequired , description to pretty table object (inPTable)
    """
    global SPACE_CHAR  # No space character that will be replaced when we print this table to text/html

    # Go ver dictionary
    for item in inDictionary:
        if isinstance(item, dict):  # If it again dictionary call same function with this new dictionary
            # inPTable.add_row([SPACE_CHAR, SPACE_CHAR, SPACE_CHAR, SPACE_CHAR])
            inPTable.add_row([SPACE_CHAR, SPACE_CHAR])
            printDic(item, inPTable, indent)
        else:
            # Two way to get next item based on input type
            if isinstance(inDictionary, dict):
                moreStuff = inDictionary.get(item)
            elif isinstance(inDictionary, list):
                # If it simple array/list we just print all it's value and we are done
                for _item in inDictionary:
                    # inPTable.add_row([indent + _item, SPACE_CHAR + SPACE_CHAR, SPACE_CHAR + SPACE_CHAR,
                    #                   listToString(get_sentences(1, True))])
                    inPTable.add_row([indent + _item, SPACE_CHAR + SPACE_CHAR])
                break

            # if it is dictionary or list process them accordingly
            if isinstance(moreStuff, dict):
                # inPTable.add_row([indent + item, SPACE_CHAR + SPACE_CHAR, SPACE_CHAR + SPACE_CHAR,
                #                   listToString(get_sentences(1, True))])
                inPTable.add_row([indent + item, SPACE_CHAR + SPACE_CHAR])
                # printDic(moreStuff, inPTable, SPACE_CHAR + SPACE_CHAR + indent)
                printDic(moreStuff, inPTable, SPACE_CHAR + SPACE_CHAR + indent)
            elif isinstance(moreStuff, list):

                # If we are not in nested call (as indent is empty string) we add one extra row in table (for clarity)
                if indent == "":
                    # inPTable.add_row([SPACE_CHAR, SPACE_CHAR, SPACE_CHAR, SPACE_CHAR])
                    inPTable.add_row([SPACE_CHAR, SPACE_CHAR])
                #
                # inPTable.add_row([indent + item, "", "", listToString(get_sentences(1, True))])
                inPTable.add_row([indent + item, ""])
                for dicInDic in moreStuff:
                    if dicInDic is not None:
                        if isinstance(dicInDic, dict):
                            # printDic(dicInDic, inPTable, SPACE_CHAR + SPACE_CHAR + SPACE_CHAR + SPACE_CHAR + indent)
                            printDic(dicInDic, inPTable,  SPACE_CHAR + SPACE_CHAR + SPACE_CHAR + SPACE_CHAR + indent)
            else:
                # Most of the call will end-up eventually here -
                # this will print - key,value,isItRequired, Lorem ipsum (description)
                inPTable.add_row(
                    [indent + item, inDictionary[item]])
                # inPTable.add_row(
                #     [indent + item, inDictionary[item], SPACE_CHAR + SPACE_CHAR, listToString(get_sentences(1, True))])


"""
    Read given yaml file 
       process each to level element build table (HTML) out of it and print it console/file 
"""
with open(INPUT_YAML) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    yaml_file_object = yaml.load(file, Loader=yaml.FullLoader)

    if PRINT_HTML:
        html_st = []
        f = open(OUTPUT_HTMl, "w")
        html_st.append(CSS_TEXT)

    i = 0
    for key in yaml_file_object:
        body_st = []
        prettyTable = PrettyTable()

        # prettyTable.field_names = ["Field", "Example Value", "Required", "Description"]
        prettyTable.field_names = ["Field", "Example Value"]

        if not PRINT_HTML:
            prettyTable.align["Field"] = "l"
            prettyTable.align["Example Value"] = "l"
            # prettyTable.align["Required"] = "c"
            # prettyTable.align["Description"] = "l"

        if isinstance(yaml_file_object, list):
            dic = yaml_file_object[i]
            i += 1
        elif isinstance(yaml_file_object, dict):
            dic = yaml_file_object.get(key)

        if isinstance(dic, dict) or isinstance(dic, list):
            printDic(dic, prettyTable, "")
            if isinstance(yaml_file_object, dict):
                yaml_snippet = yaml.dump({key: dic})
            else:
                yaml_snippet = yaml.dump(dic)

        else:
            # prettyTable.add_row([key, dic, SPACE_CHAR + SPACE_CHAR, get_sentences(1, True)[0]])
            prettyTable.add_row([key, dic])
            yaml_snippet = yaml.dump({key: dic})

        if isinstance(yaml_file_object, dict):
            if PRINT_HTML:
                body_st.append("<h2>" + key + "</h2>")
            else:
                print("=> " + key + ":")

        table = prettyTable.get_html_string(attributes={"name": key,
                                                        "id": key,
                                                        "class": "table table-striped table-condensed",
                                                        "style": "width: 1450px;table-layout: fixed;overflow-wrap: "
                                                                 "break-word;"})
        table = table.replace(SPACE_CHAR, "&nbsp;")
        body_st.append(table)
        body_st.append("Raw yaml:")
        body_st.append("<pre>" + yaml_snippet + "</pre>")

        if PRINT_HTML:
            html_st.append(" ".join(body_st))
        else:
            print(str(prettyTable).replace(SPACE_CHAR, " "))
            # print("Raw yaml:")
            # print("\t" + yaml_snippet.replace("\n", "\n\t"))

    if PRINT_HTML:
        html_st.append("</html>")
        f.write(" ".join(html_st))
        f.close()
        print("File " + OUTPUT_HTMl + " has been generated")