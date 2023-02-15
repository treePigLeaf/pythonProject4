from chatbot import Chatbot

chatbot = Chatbot()


def handle(msg):
    chat = chatbot.chat(msg)
    if chat:
        return chat
    return ""


message = "请输入你的信息!!!"

print(message)

message_ = ''

while message_ != 'quit':

    message_ = input("===========\n")
    s = handle(message_)
    response = "\033[1;31m " + s + "\033[0m"
    print(response)
    if message_ == 'quit':
        break  # 退出当前循环的命令

print('您已退出聊天')
