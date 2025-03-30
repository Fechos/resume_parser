import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)


class ResumeQA:
    def __init__(self):
        self.resume_data = None
        self.file_id = None

    def upload_resume(self, file_path):
        """上传简历文件"""
        file_object = client.files.create(
            file=Path(file_path),
            purpose="file-extract"
        )
        self.file_id = file_object.id
        return True

    def _get_resume_text(self):
        """获取简历文本内容"""
        return client.files.content(file_id=self.file_id).text

    def _extract_structured_info(self):
        """从简历中提取结构化信息"""
        resume_text = self._get_resume_text()

        prompt = f"""
        请从以下简历文本中提取结构化信息，返回一个详细的JSON对象。

        简历内容：
        {resume_text}

        请提取以下信息：
        - 基本信息（姓名、联系方式、邮箱等）
        - 教育背景（学校、专业、学历、时间等）
        - 工作经历（公司、职位、时间、工作内容等）
        - 项目经验（项目名称、角色、时间、项目描述等）
        - 技能专长（编程语言、工具、证书等）
        - 其他信息（奖项、出版物等）
        """

        response = client.chat.completions.create(
            model="moonshot-v1-32k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        try:
            self.resume_data = json.loads(response.choices[0].message.content)
        except:
            self.resume_data = {"raw_text": resume_text}

    def ask_question(self, question):
        """回答问题"""
        if not self.resume_data:
            return "请先上传简历文件"

        context = json.dumps(self.resume_data, indent=2, ensure_ascii=False)

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的HR助手，根据提供的结构化简历数据回答问题。"
            },
            {
                "role": "user",
                "content": f"简历数据:\n{context}\n\n问题: {question}"
            }
        ]

        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=messages,
                temperature=0.3,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"发生错误: {str(e)}"


# 全局问答系统实例
qa_system = ResumeQA()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 保存文件到临时目录
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # 上传并处理简历
    try:
        qa_system.upload_resume(file_path)
        qa_system._extract_structured_info()
        answer = qa_system.ask_question("Please parse this resume.")
        return jsonify({
            "message": f"Parsing results:\n{answer}",
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = qa_system.ask_question(question)
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(debug=True, port=5000)