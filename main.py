import os
import tkinter as tk
from tkinter import font as tkfont, messagebox
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import pandas as pd

class FlanT5:
    def __init__(self, model_path):
        self.model, self.tokenizer = self.load_model(model_path)

    def load_model(self, model_path):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        return model, tokenizer

    def get_answer(self, question):
        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to("cpu")
        outputs = self.model.generate(input_ids=input_ids)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

class BERT:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        return pipeline("question-answering", model=model_path)

    def get_answer(self, question, database):
        score = 0
        best_answer = ''
        for content in database:
            res = self.model(question=question, context=content)
            if res['score'] > score:
                best_answer = res['answer']
                score = res['score']
        return best_answer

class ApplicationUI:
    def __init__(self, flan_model_path, bert_model_path, csv_path):
        self.flan_t5 = FlanT5(flan_model_path)
        self.bert = BERT(bert_model_path)
        self.database = self.load_data(csv_path)
        self.setup_gui()

    def load_data(self, csv_path):
        covid_data = pd.read_csv(csv_path)
        return covid_data['context'].tolist()

    def get_answer(self, question):
        if self.model_version.get() == 1:  # Use FLAN-T5
            answer = self.flan_t5.get_answer(question)
        else:  # Use BERT
            answer = self.bert.get_answer(question, self.database)
        return f"Question: {question}\nAnswer from our model: {answer}"

    def clear_default_text(self, event):
        if self.txt_question.get("1.0", "end-1c") == self.question_text:
            self.txt_question.delete("1.0", "end-1c")
            self.txt_question.config(foreground="#000000")

    def clicked(self):
        question = self.txt_question.get("1.0", "end").strip()
        if question and question != self.question_text:
            answer = self.get_answer(question)
            self.response_text.set(answer)
            self.lbl_response.configure(text=self.response_text.get())
            self.txt_question.delete("1.0", tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter a valid question.")

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("QA System")
        self.window.geometry("1250x1200")
        self.window.configure(bg='#f7f7f7')

        font_title = tkfont.Font(family="Arial Bold", size=20)
        font_input = tkfont.Font(family="Arial", size=16)
        font_response = tkfont.Font(family="Arial", size=14)

        self.lbl_title = tk.Label(self.window, text="Question Answering System", font=font_title, bg='#f7f7f7')
        self.lbl_title.grid(column=0, row=0, columnspan=2, pady=20, padx=20, sticky='n')

        self.lbl_model = tk.Label(self.window, text="Choose your model:", font=font_input, bg='#f7f7f7')
        self.lbl_model.grid(column=0, row=1, pady=20, padx=20, sticky='e')

        self.model_version = tk.IntVar()
        self.model_version.set(1)

        self.rdio_flan = tk.Radiobutton(self.window, text="FLAN-T5", font=font_input, bg='#f7f7f7', variable=self.model_version, value=1)
        self.rdio_bert = tk.Radiobutton(self.window, text="BERT", font=font_input, bg='#f7f7f7', variable=self.model_version, value=2)
        self.rdio_flan.grid(column=1, row=1, pady=20, padx=20, sticky='w')
        self.rdio_bert.grid(column=2, row=1, pady=20, padx=20, sticky='w')

        self.question_text = "Please input your question..."
        self.txt_question = tk.Text(self.window, width=85, height=5, font=font_input, borderwidth=2, relief="solid", bg='#ffffff', wrap="word")
        self.txt_question.grid(column=0, row=3, columnspan=2, pady=20, padx=20, sticky='nsew')
        self.txt_question.insert(tk.END, self.question_text)
        self.txt_question.config(foreground="#888888")
        self.txt_question.bind("<FocusIn>", self.clear_default_text)

        self.scrollbar = tk.Scrollbar(self.window, width=15, command=self.txt_question.yview)
        self.scrollbar.grid(column=2, row=3, pady=20, sticky='ns')
        self.txt_question['yscrollcommand'] = self.scrollbar.set

        self.lbl_response = tk.Label(self.window, text="", font=font_response, bg='#f7f7f7', wraplength=1000)
        self.lbl_response.grid(column=0, row=4, columnspan=2, pady=20, padx=20, sticky='ew')
        self.lbl_response.configure(justify='left')

        self.response_text = tk.StringVar()

        self.btn = tk.Button(self.window, text="Get Answer", font=font_input, bg="#4CAF50", fg='#ffffff', borderwidth=2, relief="solid", command=self.clicked)
        self.btn.grid(column=2, row=4, pady=20, padx=20, sticky='ew')

        self.window.columnconfigure((0, 1), weight=1)
        self.window.rowconfigure((2, 3, 4, 5), weight=1)

        self.window.mainloop()

if __name__ == "__main__":
    flan_model_path = "./checkpoint"
    bert_model_path = "./Bert/model/bert-COVID-QA"
    csv_path = "Bert/database_small.csv"
    ApplicationUI(flan_model_path, bert_model_path, csv_path)

