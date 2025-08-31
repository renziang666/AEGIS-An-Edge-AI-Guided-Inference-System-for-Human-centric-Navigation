我正在做一个语音问答项目，项目原理是用ASR将用户语音提问转化为文字，结合RAG寻找到最适答案，最后统一提交到LLM生成适当的回复。在调试时遇到了一个问题，即ASR识别准确率不高，比如用户语音输入为“我预约的医生的问诊时间是什么？”，ASR有可能会识别成“那其我预约的医生的问阵时间是什么？”。这会引导llm生成不当回答。我想通过设计一套系统提示词附加到llm输入消息之前，来引导llm理解到真正正确的表述，而不是死板地按照错误的识别输入回答。请你帮我写好这份提示词。
系统提示词的消息格式请用python的字典形式表达，比如
```Python
system_prompt_str = "你是一个助手。"
{
    "role": "system",
    "content": f"""
    {system_prompt_str}
    """
}
```
每次触发回答任务时，系统提示词会和以下的用户提示词一同送入LLM处理，即
```python
RAG_result_str = "李华医生的预约时间是明天10:00"
ASR_result_str_with_some_errors = "那其我预约的医生的问阵时间是什么？"
{
    "role": "user",
    "content": f"""
        已知信息：'{RAG_result_str}'。请根据这个信息，回答问题：'{ASR_result_str_with_some_errors}'。
    """
}
```