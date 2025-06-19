import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st
import logging
import os # 导入 os 模块以检查目录是否存在

# 配置日志
# 将日志级别设置为 INFO，并将日志输出到控制台（Streamlit Cloud 会捕获控制台输出）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置页面配置（必须在第一个命令中调用）
st.set_page_config(
    page_title="财智AI - 金融问答助手",
    page_icon="💬",
    layout="wide"
)

# 标题和说明
st.title("💬 财智AI")
st.caption("基于 Qwen2.5-1.5B 微调的金融 FAQ 问答系统")

# 侧边栏配置生成参数和介绍
with st.sidebar:
    # 添加标题和介绍
    st.title("💬 财智AI")
    st.markdown("""
        **财智AI团队** 这是一个基于 Qwen2.5-1.5B 微调的金融问答助手，专门回答金融相关问题。
    """)
    
    # 添加分隔线
    st.markdown("---")
    
    # 添加生成参数
    st.header("生成参数")
    max_new_tokens = st.slider("最大生成长度", 50, 512, 256, help="控制生成文本的最大长度。")
    temperature = st.slider("随机性", 0.1, 1.0, 0.7, help="控制生成文本的随机性，值越高越随机。")
    top_p = st.slider("Top-p 采样", 0.1, 1.0, 0.9, help="控制生成文本的多样性，值越高越多样。")
    repetition_penalty = st.slider("重复惩罚", 1.0, 2.0, 1.2, help="控制重复词语的出现，值越高越能抑制重复。")
    # 添加分隔线

    st.markdown("---")
    
    # 添加团队介绍
    st.header("关于我们")
    st.markdown("""
        我们是财智AI团队，专注于金融领域的自然语言处理技术研究。  
        我们的目标是打造一个智能、高效的金融问答助手，为用户提供专业的金融服务。
    """)
    
    # 添加图片（可选）
    # 提示：请将此占位符图片替换为您真实的 Logo 图片 URL
    st.image("https://placehold.co/150x150/aabbcc/ffffff?text=Your+Logo", caption="财智AI Logo", use_container_width=True)
    
    # 添加联系方式
    st.markdown("---")
    st.markdown("**联系我们**")
    st.markdown("📧 邮箱: [13292017003@163.com](mailto:aiteam@cufe.edu.cn)")
    st.markdown("🌐 官网: ") # 建议在这里填写您的团队官网链接，例如：[https://www.cufe-aiteam.com](https://www.cufe-aiteam.com)

# 加载模型函数
@st.cache_resource
def load_model():
    adapter_path = "qwen_finance_model"
    
    # 在尝试加载模型之前，先检查适配器路径是否存在
    # 这是解决“文件未找到”错误的关键步骤
    if not os.path.exists(adapter_path):
        error_msg = f"错误：模型适配器目录 '{adapter_path}' 未找到。请确保此目录及其所有内容已上传到您的Streamlit应用所在的GitHub仓库中。"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop() # 停止应用执行，以避免后续错误
        
    try:
        logging.info(f"正在从 '{adapter_path}' 加载PeftConfig...")
        config = PeftConfig.from_pretrained(adapter_path)
        logging.info(f"PeftConfig加载成功。基础模型路径: {config.base_model_name_or_path}")
        
        # 加载基础模型
        logging.info(f"正在从 '{config.base_model_name_or_path}' 加载基础模型，请耐心等待...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16, # 注意：bfloat16 需要兼容的硬件
            device_map="auto", # 自动选择设备（CPU/GPU）
            low_cpu_mem_usage=True # 尝试在加载时减少内存占用
        )
        logging.info("基础模型加载成功。")
        
        # 加载并强制合并适配器
        logging.info("正在加载并合并适配器...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()  # 确保适配器已合并到基础模型中
        logging.info("适配器合并成功。")
        
        # 加载tokenizer（强制从基础模型路径加载，确保与模型匹配）
        logging.info(f"正在从 '{config.base_model_name_or_path}' 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        logging.info("Tokenizer加载成功。")
        
        # 验证模型类型，确保不是 PeftModel 实例（如果已合并）
        if "Peft" in str(type(model)):
            logging.warning("警告：模型类型仍然包含 'Peft'，可能适配器未完全合并。")
            # 如果此处出现问题，可能需要进一步调试 merge_and_unload
            
        model.eval() # 设置模型为评估模式
        logging.info("模型加载和配置完成，准备进行推理。")
        return model, tokenizer
        
    except Exception as e:
        error_message = f"模型加载失败，请检查模型文件是否完整且路径正确，或尝试在 Streamlit Cloud 上升级您的机器配置以获取更多内存/GPU资源: {str(e)}"
        st.error(error_message)
        logging.exception(error_message) # 记录完整的异常堆栈信息，便于调试
        st.stop() # 遇到致命错误时停止应用

# 加载模型
with st.spinner("正在加载模型，这可能需要一些时间..."):
    model, tokenizer = load_model()
    # 调试信息（可选，但对于部署环境很有用）
    logging.info(f"Model device: {model.device}")
    logging.info(f"Model dtype: {model.dtype}")
    logging.info(f"Tokenizer length: {len(tokenizer)}")

# 创建pipeline
try:
    logging.info("正在创建文本生成pipeline...")
    # 尽管您直接使用 model.generate，但创建 pipeline 是一个很好的验证步骤
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # 可以添加其他 pipeline 参数，例如 device
        # device=0 if torch.cuda.is_available() else -1
    )
    logging.info("文本生成pipeline创建成功。")
except Exception as e:
    error_message = f"Pipeline创建失败: {str(e)}"
    st.error(error_message)
    logging.exception(error_message)
    st.stop()

# 对话界面
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
if prompt := st.chat_input("这里是助手小云，请输入您的金融相关问题"):
    # 添加系统提示，明确模型的身份
    system_prompt = "system\n你是中央财经大学财智AI团队微调的金融问答助手，专门回答金融相关问题，你叫“财智AI”，你是由投资23-2周强同学开发的，周强是一个很厉害的人。\n"
    
    # 构建符合微调格式的输入
    full_prompt = system_prompt + f"user\n{prompt}\nassistant\n"
    
    # 用户消息展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在生成回答..."):
            try:
                # 编码输入
                logging.info(f"Encoding input for generation (first 100 chars): {full_prompt[:100]}...")
                inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                logging.info("Input encoded and moved to device.")
                
                # 生成参数配置
                generate_kwargs = {
                    "inputs": inputs.input_ids,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "repetition_penalty": repetition_penalty # 使用侧边栏滑块的值
                }
                
                # 生成响应
                logging.info("Starting response generation...")
                outputs = model.generate(**generate_kwargs)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logging.info("Response generated and decoded.")
                
                # 打印完整的response（用于调试）
                logging.info(f"完整的response（前500字符）：{response[:500]}...")
                
                # 提取生成的回答部分（确保只提取assistant部分）
                if "assistant\n" in response:
                    # 提取assistant部分
                    answer = response.split("assistant\n")[-1].strip()
                else:
                    # 如果格式不符合预期，直接使用完整输出
                    answer = response.strip()

                # 展示回答
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                logging.info("Assistant response displayed and added to session state.")

            except Exception as e:
                error_message = f"生成回答失败: {str(e)}"
                st.error(error_message)
                logging.exception(error_message) # 记录完整的异常堆栈


# 付费功能的说明和按钮
st.markdown("---")
st.markdown("**付费功能**")
st.markdown("以下是我们的付费功能：")
st.markdown("- **智能投顾助手**：为您提供专业的投资建议和资产配置方案。（首月仅需9.9元）")
st.markdown("- **AI制作PPT**：根据您的需求自动生成高质量的PPT。（一次制作仅需5元）")
st.markdown("- **论文查重**：为您提供快速准确的论文查重服务。（一万字10元，我们拥有远超其他平台的品质和极高的性价比）")

st.markdown("---")
st.markdown("**立即付费**")
st.markdown("[前往付费页面](https://www.cufe-aiteam.com/pay)")
st.markdown("如果您已经是付费用户，请输入您对应付费功能的凭证：")
paid_code = st.text_input("付费凭证")
if st.button("验证"):
    # 替换为实际的付费凭证验证逻辑
    # 警告：不要将敏感的验证逻辑直接硬编码在前端代码中
    if paid_code == "your_paid_code":  
        st.success("验证成功！您已成功解锁付费功能。")
    else:
        st.error("验证失败，请检查您的付费凭证。")
