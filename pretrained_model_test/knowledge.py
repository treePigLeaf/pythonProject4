import requests
from transformers import pipeline


class Knowledge:

    def __init__(self, device="cpu") -> None:
        self.ner = pipeline(
            model="ckiplab/bert-base-chinese-ner", aggregation_strategy="simple", framework="pt", device=device)

    def get_entities(self, s):
        results = []
        prev_end = None
        prev_word = ""
        for entity in self.ner(s):
            start = entity["start"]
            end = entity["end"]
            word = entity["word"]
            if prev_end is not None and start != prev_end:
                results.append(prev_word.replace(" ", ""))
                prev_word = ""
            prev_word += word
            prev_end = end
        if len(prev_word):
            results.append(prev_word.replace(" ", ""))
        return results

    def get_knowledge_text(self, s, topk=1):
        knowledges = []
        entities = self.get_entities(s)
        print("\033[1;35m 提取的实体列表\033[0m")
        print("\033[1;36m   ---------------\033[0m")
        print(entities)
        print("\033[1;36m   ---------------\033[0m")
        for entity in entities:
            resp = requests.get("http://shuyantech.com/api/cndbpedia/avpair", params={"q": entity})
            ret_ = resp.json()["ret"]
            if len(ret_) > 0:
                for e in ret_:
                    knowledges.extend(e[0])
                    knowledges.extend(':')
                    knowledges.extend(e[1])
                    knowledges.extend(';')
            else:
                resp = requests.get("https://api.ownthink.com/kg/knowledge", params={"entity": entity})
                jj = resp.json()['data']
                entity_ = jj['entity']
                entity_ = entity_.replace(" ", "")
                knowledges.extend(entity_)
                desc_ = jj['desc']
                desc_ = desc_.replace(" ", "")
                knowledges.extend(desc_)
                for ee in jj['avp']:
                    knowledges.extend(ee[0])
                    knowledges.extend(":")
                    knowledges.extend(ee[1])
                    knowledges.extend(';')
        return "".join(knowledges)
