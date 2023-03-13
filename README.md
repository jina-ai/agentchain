# AgentChain 
AgentChain uses Large Language Models (LLMs) for reasoning and orchestrating multiple LLMs or Large Models (LMs) for accomplishing sophisticated tasks. AgentChain is fully multimodal: it accepts text, image, audio, webcam, tabular data as input and output.

# Demo

# Examples

### Example 1: 🏝️📸🌅 AgentChain Image Generation System for Travel Company
As a travel company that is promoting a new and exotic destination, it is crucial to have high-quality images that can grab the attention of potential travelers. However, manually creating stunning images can be time-consuming and expensive. That's why the travel company wants to use AgentChain to automate the image generation process and create beautiful visuals with the help of various agents.

Here is how AgentChain can help by chaining different agents together:
1. Use `SearchAgent` (`Google Search API`, `Wikipedia API`, `Serp`) to gather information and inspiration about the destination, such as the most popular landmarks, the local cuisine, and the unique features of the location.
2. Use `ImageAgent` (`Upscaler`) to enhance the quality of images and make them more appealing by using state-of-the-art algorithms to increase the resolution and remove noise from the images.
3. Use `MultiModalAgent` (`OpenAI Whisper`) to generate descriptive captions for the images, providing more context and making the images more meaningful.
4. Use `CommsAgent` (`TwilloEmailWriter`) to send the images to the target audience via email or other messaging platforms, attracting potential travelers with stunning visuals and promoting the new destination.

### Example 2: 💼💹📈 AgentChain Financial Analysis Report for Investment Firm
As an investment firm that manages a large portfolio of stocks, it is critical to stay up-to-date with the latest market trends and analyze the performance of different stocks to make informed investment decisions. However, analyzing data from multiple sources can be time-consuming and error-prone. That's why the investment firm wants to use AgentChain to automate the analysis process and generate reports with the help of various agents.

Here is how AgentChain can help by chaining different agents together:
1. Use `ToolsAgent` (`Python REPL`) to analyze data from different sources (e.g., CSV files, stock market APIs) and perform calculations related to financial metrics such as earnings, dividends, and P/E ratios.
2. Use `SearchAgent` (`Bing API`) to gather news and information related to the stocks in the portfolio, such as recent earnings reports, industry trends, and analyst ratings.
3. Use `MultiModalAgent` (`ControlNet`) to perform sentiment analysis on the news and information gathered, providing insights into market sentiment and potential trends.
4. Use `CommsAgent` (`TwilloEmailWriter`) to send a summary report of the analysis to the appropriate stakeholders, helping them make informed decisions about their investments.

### Example 3: 🛍️💬💻 AgentChain Customer Service Chatbot for E-commerce Site
As an e-commerce site that wants to provide excellent customer service, it is crucial to have a chatbot that can handle customer inquiries and support requests in a timely and efficient manner. However, building a chatbot that can understand and respond to complex customer requests can be challenging. That's why the e-commerce site wants to use AgentChain to automate the chatbot process and provide superior customer service with the help of various agents.

Here is how AgentChain can help by chaining different agents together:
1. Use `MultiModalAgent` (`Blip2`) to handle input from various modalities (text, image, audio, webcam), making it easier for customers to ask questions and make requests in a natural way.
2. Use `SearchAgent` (`Google Search API`, `Wikipedia API`) to provide information about products or services, such as specifications, pricing, and availability.
3. Use `CommsAgent` (`TwilloMessenger`) to communicate with customers via messaging platforms, providing support and answering questions in real-time.
4. Use `ToolsAgent` (`Math`) to perform calculations related to discounts, taxes, or shipping costs, helping customers make informed decisions about their purchases.
5. Use `MultiModalAgent` (`Coqui`) to generate natural-sounding responses and hold more complex conversations, providing a personalized and engaging experience for customers.

### Example 4: 🧑‍⚕️💊💤 AgentChain Personal Health Assistant for Aging Population
As an organization that provides support for seniors who want to live independently, it is essential to have a personal health assistant that can help seniors manage their health and well-being. However, providing personalized health advice and reminders can be challenging, especially for seniors who may have different health needs and preferences. That's why the organization wants to use AgentChain to automate the health assistant process and provide personalized support with the help of various agents.

Here is how AgentChain can help by chaining different agents together:
1. Use `MultiModalAgent` (`StableDiffusion`) to handle input from various health monitoring devices (e.g., heart rate monitors, blood pressure monitors, sleep trackers), providing real-time health data and alerts to the health assistant.
2. Use `SearchAgent` (`Google Search API`, `Wikipedia API`) to provide information about health topics and medications, such as side effects, dosage, and interactions.
3. Use `ToolsAgent` (`Python REPL`) to perform calculations related to medication dosages or nutritional requirements, providing personalized advice and reminders to seniors.
4. Use `MultiModalAgent` (`ControlNet`) to generate personalized recommendations for diet, exercise, and medication, taking into account the seniors' health goals and preferences.
5. Use `CommsAgent` (`TwilloCaller`) to make reminders and provide alerts to help seniors stay on track with their health goals, improving their quality of life and reducing the need for emergency care.




# Features

* **Multimodal input**: AgentChain allows users to input data in a variety of formats, including images, audio, text, and even tables. This makes it easy to work with a wide range of data types and sources.
* **Multimodal Output**: AgentChain is capable of delivering results in a variety of formats as well, including audio, text, and image. This means users can choose the format that works best for their needs and preferences.
* **Multi-tasking**: AgentChain is designed to execute a wide range of tasks based on the input it receives. This includes everything from image recognition and text analysis to data processing and more.
* **Real-time internet search**: AgentChain provides access to real-time internet search capabilities, allowing users to quickly find and retrieve relevant information from the web. This can be especially useful for research and other knowledge-based tasks.
* **Access to multiple useful tools**: AgentChain has access to multiple tools such as Python Repl, Terminal and Math modules to create sophisticated answers and verify code.
* **Customizable**: If you need a tool you can easily configure your tool and add it to the pipeline. 


# Architecture


# Get started


## Acknowledgement
We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
[CLIPSeg](https://github.com/timojl/clipseg) &#8194;
[BLIP](https://github.com/salesforce/BLIP) &#8194;
[Microsoft](https://github.com/microsoft/visual-chatgpt) &#8194;


