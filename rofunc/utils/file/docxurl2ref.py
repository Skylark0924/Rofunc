# import numpy as np
# from docx import Document
# from docx.opc.constants import RELATIONSHIP_TYPE as RT
# from docx.oxml.ns import qn
# import zipfile
# from bs4 import BeautifulSoup
#
#
# def iter_hyperlink_rels(rels):
#     for rel in rels:
#         if rels[rel].reltype == 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink':
#             yield rels[rel]._target
#
#
# def find_hyperlink_indoc(doc):
#     '''
#     :param doc: doc file get by doc = Document('./xxxx.docx')
#     :return: a list of all hyperlink item in doc.
#     '''
#     xml_e = doc.element
#     hyperlink_list = xml_e.findall('.//' + qn("w:hyperlink"))
#     return hyperlink_list
#
#
# def get_hyperlink_text(hyperlink_item):
#     text = hyperlink_item.findall('.//' + qn("w:t"))[0].text
#     if text[0] == '[':
#         text = text.split('[')[1].split(']')[0]
#     return text
#
#
# def get_linked_text(soup):
#     links = []
#
#     # This kind of link has a corresponding URL in the _rel file.
#     for tag in soup.find_all("hyperlink"):
#         # try/except because some hyperlinks have no id.
#         try:
#             links.append({"id": tag["r:id"], "text": tag.text})
#         except:
#             pass
#     return links
#
#
# if __name__ == '__main__':
#     file_name = "D:\\Laboratory\\CUHK\\OneDrive - The Chinese University of Hong Kong\\Documents\\申报书\\擦桌子\\220222CRF1P_Xi_20220720.docx"
#     document = Document(file_name)
#
#     hl_list = find_hyperlink_indoc(document)
#     i = 0
#     text_lst = []
#     for item in hl_list:
#         i += 1
#         print(i, get_hyperlink_text(item))
#         text_lst.append(get_hyperlink_text(item))
#
#     archive = zipfile.ZipFile(file_name, "r")
#     file_data = archive.read("word/document.xml")
#     doc_soup = BeautifulSoup(file_data, "xml")
#     linked_text = get_linked_text(doc_soup)
#
#     rels = document.part.rels
#     i = 0
#     rel_lst = []
#     for rel in rels:
#         if rels[rel].reltype == RT.HYPERLINK:
#             i += 1
#             # print(i, rels[rel]._target)
#             rel_lst.append(rels[rel]._target)
#             for item in linked_text:
#                 if item['id'] == rel:
#                     item['url'] = rels[rel]._target
#
#     rel_text = np.array([[linked_text[j]['text'].split('[')[1].split(']')[0], linked_text[j]['url']] if
#                          linked_text[j]['text'][0] == '[' else [linked_text[j]['text'], linked_text[j]['url']] for j in
#                         range(len(linked_text))])
#     # rel_text = np.hstack((np.array(text_lst).reshape((-1, 1)), np.array(rel_lst).reshape((-1, 1))))
#     np.savetxt("rel_text.csv", rel_text, delimiter=",", fmt="%s")
