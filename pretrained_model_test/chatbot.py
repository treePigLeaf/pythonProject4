from transformers import BertTokenizer, BartForConditionalGeneration
from knowledge import Knowledge


class Chatbot:

    def __init__(self, use_kg=True, device="cpu"):
        if use_kg:
            self.knowledge = Knowledge(device)
        self.tokenizer = BertTokenizer.from_pretrained("HIT-TMG/dialogue-bart-large-chinese")
        self.model = BartForConditionalGeneration.from_pretrained("HIT-TMG/dialogue-bart-large-chinese")
        self.model.to(device)
        self.use_kg = use_kg
        self.device = device

    def chat(self, s):
        if self.use_kg:
            k = self.knowledge.get_knowledge_text(s)
            print("\033[1;33m 知识图谱返回的结果\033[0m")
            print("\033[1;34m   ---------------\033[0m")
            print(k)
            print("\033[1;34m   ---------------\033[0m")
            input_ids = self.tokenizer("对话历史：" + s, "知识：" + k, return_tensors="pt", truncation=True,
                                       max_length=512).input_ids
        else:
            input_ids = self.tokenizer("对话历史：" + s, return_tensors="pt", truncation=True, max_length=512).input_ids
        outputs = self.model.generate(input_ids.to(self.device), max_length=100, top_p=0.5, do_sample=True)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text.replace(" ", "")
        return output_text


if __name__ == '__main__':
    chat = Chatbot()
    history = "可以在国图使用别人的读者卡吗"
    history_str = "对话历史：" + history
    knowledge = "周杰伦[华语流行乐男歌手、音乐人、演员、导演、编剧]周杰伦（JayChou），1979年1月18日出生于台湾省新北市，祖籍福建省泉州市永春县，中国台湾流行乐男歌手、音乐人、演员、导演、编剧，毕业于淡江中学。主要作品:叱咤风云;主要作品:练爱ING;主要作品:极限特工4;主要作品:惊天魔盗团2;主要作品:天台爱情;主要作品:逆战;主要作品:青蜂侠;主要作品:苏乞儿;主要作品:刺陵;主要作品:大灌篮;主要作品:不能说的秘密;主要作品:满城尽带黄金甲;主要作品:头文字D;主要作品:自行我路;主要作品:寻找周杰伦;主要作品:熊猫人;主要作品:蓝星;主要作品:星情花园;主要作品:百里香煎鱼;"
    # knowledge = ["图书馆的开放时间是工作日的周一到周五的8点到17点", "图书馆藏书有18000本"]
    knowledge = "浙江大学简称浙大，创办时间:1897年05月21日，有7家附属医院，设有紫金港、玉泉、西溪、华家池、之江、舟山、海宁、宁波等8个校区，浙江大学设有7个学部、37个专业学院（系）、1个工程师学院、2个中外合作办学学院，浙江大学占地面积6223440平方米。浙江大学的创建人叫二蛋，他生于1990年"
    knowledge = "国家图书馆读者卡和少年儿童馆读者卡只限本人使用"
    knowledge_str = "知识：" + knowledge
    te = history_str  # + knowledge_str
    input_ids = chat.tokenizer(history_str+knowledge_str, return_tensors="pt", truncation=True,
                               max_length=512).input_ids
    output_ids = chat.model.generate(input_ids.to(chat.device), max_length=100, top_p=0.8, do_sample=True)
    decode = chat.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    decode = decode.replace(" ", "")
    print(decode)
