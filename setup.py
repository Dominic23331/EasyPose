from setuptools import setup, find_packages

# 定义包名称、版本号等信息
name = 'easypose'
version = '0.1.0'
description = 'EasyPose is a human pose estimation algorithm library ' \
              'that can call multiple algorithms through simple functions'
author = 'Yongtao Wang'
url = 'https://github.com/Dominic23331/EasyPose'

# 设置依赖项（可选）
install_requires = [
    'numpy>=1.24.4',
    'opencv-python',
    'tqdm'
]

# 使用setuptools库进行打包配置
setup(
    name=name,
    version=version,
    description=description,
    author=author,
    url=url,
    packages=find_packages(),
    install_requires=install_requires,
)
