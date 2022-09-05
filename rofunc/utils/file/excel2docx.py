import pandas as pd
import docx


def csv2docx(csv_file, docx_out_path):
    df = pd.read_csv(csv_file, encoding="gbk")
    data = df.iloc[:, [0, 2]].values

    docx_file = docx.Document()
    for i in range(len(data)):
        docx_file.add_paragraph('[{}] {}'.format(data[i, 0], data[i, 1]))
    docx_file.save(docx_out_path)

if __name__ == '__main__':
    csv_fname = 'D:\\Github\\Knowledge-Universe\\Robotics\\Roadmap-for-robot-science\\Rofuncs\\python\\file_process\\rel_text.csv'
    docx_out_path = "D:\\Laboratory\\CUHK\\OneDrive - The Chinese University of Hong Kong\\Documents\\申报书\\擦桌子\\Reference.docx"
    csv2docx(csv_fname, docx_out_path)
