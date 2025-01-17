
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    xml_path=path + '\\*.xml'
    for xml_file in glob.glob(xml_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,
                    int(root.find('size').find('width').text),
                    int(root.find('size').find('width').text),
                    member.find('name').text,
                    int(member.find('bndbox').find('xmin').text),
                    int(member.find('bndbox').find('ymin').text),
                    int(member.find('bndbox').find('xmax').text),
                    int(member.find('bndbox').find('ymax').text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'train\\annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('road_signs_labels.csv', index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()