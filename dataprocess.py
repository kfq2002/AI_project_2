# coding=utf-8
import os, random, shutil

# 将图片拆分成训练集train(0.8)和验证集val(0.2)

def moveFile(Dir,val_ratio=0.5,test_ratio=0.5):
    path = 'AI_project_2/data/test'
    now_dir = os.path.split(Dir)[1]
    if not os.path.exists(os.path.join(path, now_dir)):
        os.makedirs(os.path.join(path, os.path.split(Dir)[1]))

    filenames = []
    for root,dirs,files in os.walk(Dir):
        for name in files:
            filenames.append(name)
        break
    
    filenum = len(filenames)

    num_test = int(filenum * test_ratio)
    sample_test = random.sample(filenames, num_test)

    for name in sample_test:
        shutil.move(os.path.join(Dir, name), os.path.join(path, now_dir))
        #os.unlink(os.path.join(Dir, name))


if __name__ == '__main__':
    Dir = 'AI_project_2/data/val'
    for root,dirs,files in os.walk(Dir):
        for name in dirs:
            folder = os.path.join(root, name)
            print("正在处理:" + folder)
            moveFile(folder)
        print("处理完成")
        break