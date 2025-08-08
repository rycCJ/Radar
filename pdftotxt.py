import PyPDF2
import os


def pdf_to_txt(pdf_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("指定的PDF文件未找到")
        # 检查文件是否为PDF
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("文件必须是PDF格式")

        # 获取文件名(不含扩展名)
        file_name = os.path.splitext(pdf_path)[0]
        # 创建输出txt文件路径
        txt_path = f"{file_name}.txt"

        # 打开PDF文件
        with open(pdf_path, 'rb') as pdf_file:
            # 创建PDF阅读器对象
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # 获取PDF页数
            num_pages = len(pdf_reader.pages)
            # 初始化存储提取文本的字符串
            words = []

            # 逐页提取文字
            for page_num in range(num_pages):
                # 获取页面对象
                page = pdf_reader.pages[page_num]
                # 提取文字
                page_text = page.extract_text()
                if page_text:
                    # 按空格分割成单词
                    page_words = page_text.split()
                    words.extend(page_words)

        # 将提取的单词写入txt文件，每行一个单词
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for word in words:
                txt_file.write(word + '\n')

        print(f"\n成功提取 {num_pages} 页内容!")
        print(f"单词已保存到: {txt_path}")
        return True
    except FileNotFoundError as e:
        print(f"\n错误: {str(e)}")
        return False
    except ValueError as e:
        print(f"\n错误: {str(e)}")
        return False
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        return False


def main():
    print("欢迎使用PDF单词提取工具!")
    print("请输入完整的PDF文件路径(或输入'q'退出)")
    while True:
        # 获取用户输入
        pdf_path = input("\nPDF文件路径: ").strip()
        # 检查是否退出
        if pdf_path.lower() == 'q':
            print("程序已退出")
            break
        # 执行转换
        success = pdf_to_txt(pdf_path)
        if success:
            while True:
                choice = input("\n是否继续处理其他文件?(y/n): ").lower().strip()
                if choice in ('y', 'n'):
                    break
                print("请输入'y'或'n'")
            if choice == 'n':
                print("程序已退出")
                break
        else:
            print("请检查文件路径后重试")


if __name__ == "__main__":
    main()