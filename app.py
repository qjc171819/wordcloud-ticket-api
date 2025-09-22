import matplotlib
matplotlib.use('Agg')  # 必须在导入 plt 前设置！
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
import pandas as pd
import jieba
import re
import jieba.posseg as pseg
from collections import Counter
from wordcloud import WordCloud
import numpy as np
from io import BytesIO
import base64
from datetime import datetime
import os
import logging
import requests
import json


app = Flask(__name__)

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# base64图片转url
def generate_url(base64_image):
    postUrl = r'https://api.imgbb.com/1/upload'
    api_key = 'cd252b3a315af679db9b6f10dbe1eff9'
    # user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0'
    req = requests.post(f'{postUrl}?key={api_key}', data = {'image': base64_image}
                        #,headers = {'user-agent': user_agent}
                        )
    js = req.json()
    image_url = js['data']['image']['url']
    return image_url


# 初始化jieba分词器（只在启动时执行一次）
def init_jieba():
    # 设备/零件前缀词库
    DEVICE_PREFIX = ["转进头", "油压机", "转一", "转二", "马达", "吸嘴", "芯片", "框架", "封装",
                     "焊片", "炉子", "模具", "皮带", "链条", "直震", "圆振", "引直", "极测",
                     "封一", "封二", "轨一", "轨二", "测一", "测二", "测试", "影像", "印字", "盖带",
                     "设备", "焊炉", "镭射", "拉刀", "夹子", "顶针", "气缸", "阀", "台", "机", "炉", "锡", "锡点", "锡膏"]

    # 添加设备相关词汇
    for prefix in DEVICE_PREFIX:
        jieba.add_word(prefix, freq=1000)

    # 添加否定性复合词
    negative_terms = [
        "不合格", "不良率", "不良品", "不工作", "不转动", "不推料", "不下料", "不焊",
        "未焊住", "未检出", "未测试", "未达标", "未完成", "未通过", "无效果", "无响应",
        "无法启动", "无法运行", "无法测试", "无法检出", "异常报警", "异常停机", "异常高",
        "异常低", "异常波动", "异常偏高", "异常偏低"
    ]
    for term in negative_terms:
        jieba.add_word(term, freq=1000)

    # 添加复合故障词
    faults = ["断脚", "漏油", "漏晶", "卡料", "压伤", "死机", "翘脚", "报警频繁", "吸不起", "不下料",
              "缺角", "印不到", "印字深浅", "偏位", "竖芯片", "粘连", "堵料", "反极性", "打弯管", "测不到",
              "短路率", "本体破损", "盖带断裂", "不转动", "掉锡", "刮花", "不推料", "不合模", "氧化", "压痕浅",
              "错位", "未焊住", "虚焊", "脱焊", "侧立", "反贴", "假焊", "冷焊"]
    for fault in faults:
        jieba.add_word(fault, freq=1000)

    # 电性不良相关词
    elec_faults = ["IR不良", "VF不良", "VB不良", "CP超标", "AP不良", "O/S不良",
                   "漏电流", "耐压不足", "电性不良", "参数超标"]
    for fault in elec_faults:
        jieba.add_word(fault, freq=1000)


# 执行初始化
init_jieba()


# 高级文本清洗函数
def clean_text(text):
    # 移除HTML标签、数字、英文、特殊符号
    text = re.sub(r'<[^>]+>', '', text)  # HTML标签
    text = re.sub(r'[a-zA-Z0-9\s+\.\%\/\、\°\&\;\-]+', ' ', text)  # 数字和英文
    text = re.sub(r'[\W]', ' ', text)  # 特殊符号
    text = re.sub(r'\s+', ' ', text)  # 多个空格
    return text.strip()


# 精准分词与词性标注函数
def precise_cut(text):
    words = pseg.cut(text)
    return words


# 智能过滤规则 - 专注不合格问题
def is_valid_term(word_obj):
    word, flag = word_obj.word, word_obj.flag

    # 过滤正面表述和合格相关词
    positive_terms = ["达标", "正常", "OK", "通过", "良品", "Pass"]
    if any(pt in word for pt in positive_terms):
        return None

    # 过滤计量/单位词
    units = ["Kpcs", "K", "pcs", "kg", "%", "mv", "V", "A", "mA", "Ω", "℃"]
    if word in units:
        return None

    # 过滤中性词
    neutral_terms = ["调试", "设置", "操作", "调整", "显示", "运行", "需要"]
    if word in neutral_terms:
        return None

    # 良率问题统一归并
    if "良率" in word and "不良率" not in word:
        return "良率问题"
    if "良率" in word:
        return "良率问题"

    # 电性不良问题归并
    if re.search(r'(IR|VF|VB|CP|AP|O/S).*(不良|大|小|高|低|超标)', word):
        return re.sub(r'(合格|正常|Pass)', '不良', word)

    # 保留复合技术问题
    DEVICE_PREFIX = ["转进头", "油压机", "转一", "转二", "马达", "吸嘴", "芯片", "框架", "封装",
                     "焊片", "炉子", "模具", "皮带", "链条", "直震", "圆振", "引直", "极测",
                     "封一", "封二", "轨一", "轨二", "测一", "测二", "测试", "影像", "印字", "盖带",
                     "设备", "焊炉", "镭射", "拉刀", "夹子", "顶针", "气缸", "阀", "台", "机", "炉", "锡", "锡点", "锡膏"]

    if word in DEVICE_PREFIX or any(word.startswith(p) for p in DEVICE_PREFIX):
        return word

    # 保留特定技术故障词
    tech_faults = ["死机", "漏油", "卡料", "断脚", "压伤", "翘脚", "吸不起",
                   "测不到", "短路率", "堵料", "反极性", "氧化", "压痕浅", "未焊住",
                   "虚焊", "脱焊", "假焊", "冷焊", "侧立", "反贴"]
    if word in tech_faults:
        return word

    # 保留否定性复合词
    if word in ["不合格", "不良率", "不良品", "不工作", "不转动", "不推料", "不下料"]:
        return word

    # 保留长度在2-4字的名词性故障描述
    if 2 <= len(word) <= 4 and flag.startswith('n'):
        return word

    return None


# 生成自定义形状词云
def generate_custom_wordcloud(word_freq):
    try:
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(current_dir, 'fonts', 'msyh.ttc')

        # 检查字体文件是否存在
        if not os.path.exists(font_path):
            # 尝试备用路径
            font_path = '/usr/share/fonts/truetype/microsoft/microsoft-yahei.ttf'
            if not os.path.exists(font_path):
                # 最后尝试Windows路径
                font_path = 'C:/Windows/Fonts/msyh.ttc'
                if not os.path.exists(font_path):
                    raise FileNotFoundError("中文字体文件缺失，请确保msyh.ttc存在于fonts目录")

        logger.info(f"使用字体路径: {font_path}")

        # 配置词云
        wc = WordCloud(
            font_path=font_path,
            background_color='white',
            max_words=50,
            colormap='Reds',
            contour_width=1,
            contour_color='#1f77b4',
            scale=2,
            random_state=42,
            width=600,
            height=400,
            margin=5
        )

        # 生成词云
        wc.generate_from_frequencies(word_freq)

        # 创建图像缓冲区
        img_buffer = BytesIO()
        plt.figure(figsize=(6, 4), dpi=100)
        plt.imshow(wc, interpolation='lanczos')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        img_buffer.seek(0)
        return img_buffer

    except Exception as e:
        logger.error(f"生成词云时出错: {str(e)}")
        raise


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        # 接收JSON数据
        data = request.json

        # 验证输入数据
        if not data or 'records' not in data:
            logger.warning("无效的请求数据: 缺少'records'字段")
            return jsonify({'error': '无效的请求数据'}), 400

        # 创建数据帧
        records = data['records'][0]['entity']['Power BI values']
        dataset = pd.DataFrame(records)

        # 检查必要字段
        if '异常描述' not in dataset.columns:
            logger.warning("数据中缺少'异常描述'列")
            return jsonify({'error': '数据中缺少"异常描述"列'}), 400

        # 获取异常描述列的所有文本
        text = '\n'.join(dataset['异常描述'].astype(str).tolist())

        # 如果文本为空
        if not text.strip():
            logger.warning("'异常描述'列内容为空")
            return jsonify({'error': '异常描述内容为空'}), 400

        # 清洗文本
        cleaned_text = clean_text(text)
        logger.info(f"清洗后文本长度: {len(cleaned_text)}")

        # 执行分词与词性标注
        word_objects = precise_cut(cleaned_text)

        # 过滤处理
        filtered_terms = []
        for wo in word_objects:
            result = is_valid_term(wo)
            if result == "良率问题":  # 特殊处理良率问题
                filtered_terms.append("良率问题")
            elif result:  # 其他有效术语
                filtered_terms.append(result)

        logger.info(f"过滤后术语数量: {len(filtered_terms)}")

        # 构建复合词（设备+故障）
        DEVICE_PREFIX = ["转进头", "油压机", "转一", "转二", "马达", "吸嘴", "芯片", "框架", "封装",
                         "焊片", "炉子", "模具", "皮带", "链条", "直震", "圆振", "引直", "极测",
                         "封一", "封二", "轨一", "轨二", "测一", "测二", "测试", "影像", "印字", "盖带",
                         "设备", "焊炉", "镭射", "拉刀", "夹子", "顶针", "气缸", "阀", "台", "机", "炉", "锡", "锡点",
                         "锡膏"]

        compound_terms = []
        for i in range(len(filtered_terms) - 1):
            # 设备名称+故障现象的组合 - 添加额外检查
            if (filtered_terms[i] in DEVICE_PREFIX and
                    filtered_terms[i + 1] not in DEVICE_PREFIX and
                    "良率问题" not in filtered_terms[i + 1] and
                    not filtered_terms[i + 1].endswith(('问题', '故障'))):
                compound_terms.append(filtered_terms[i] + filtered_terms[i + 1])

        # 词频统计
        term_counts = Counter(compound_terms + filtered_terms)
        logger.info(f"高频词汇: {term_counts.most_common(10)}")

        # 生成词云图像
        image_buffer = generate_custom_wordcloud(term_counts)

        # 将图像转换为Base64字符串 - 这是更可靠的解决方案
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        # 将Base64转成图片URL
        imageUrl = generate_url(image_base64)

        # 计算 ticket_type
        ticket_types = set(dataset['工单类型'])
        if {'S', 'M', 'T'}.issubset(ticket_types):
            ticket_type = "Overview"
        elif 'S' in ticket_types:
            ticket_type = "S"
        elif 'M' in ticket_types:
            ticket_type = "M"
        elif 'T' in ticket_types:
            ticket_type = "T"
        else:
            ticket_type = "Unknown"

        # 返回结果（包含Base64图像和词频数据）
        return jsonify({
            'image_base64': image_base64,
            'word_freq': term_counts.most_common(20),
            'status': 'success',
            'ticket_type': ticket_type,
            'image_Url': imageUrl,
            'created_at': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"API处理错误: {str(e)}", exc_info=True)
        return jsonify({
            'error': '处理过程中发生错误',
            'details': str(e)
        }), 500


# 移除cleanup路由 - 使用Base64后不再需要临时文件

if __name__ == '__main__':
    # 生产环境应设置debug=False
    app.run(host='0.0.0.0', port=5080, debug=False)





