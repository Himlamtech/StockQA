# Stock Knowledge Q&A System

## Overview

This is an AI-powered question-answering system for Vietnamese stock market information using the [vnstock](https://github.com/thinh-vu/vnstock) library. The application uses OpenAI's function calling capabilities to analyze user queries, determine the appropriate vnstock functions to call, and generate comprehensive responses based on real-time stock data.

## Features

- Natural language interface to query Vietnamese stock market data
- Automatic extraction of relevant parameters from user queries
- Dynamic data visualization of stock prices and financial metrics
- Comprehensive financial analysis of Vietnamese companies
- Chat history to track your previous queries and responses

## How It Works

1. **User Input Analysis**: When a user enters a question, OpenAI's function calling API analyzes the query to determine which vnstock function to use and what parameters to extract.

2. **Data Retrieval**: The system calls the appropriate vnstock function with the extracted parameters to retrieve the relevant data.

3. **Data Processing**: The retrieved data is processed and formatted for visualization and analysis.

4. **Response Generation**: OpenAI generates a comprehensive analysis and response based on the data and the original query.

5. **Display**: The system displays both the raw data (in charts and tables) and the natural language analysis to the user.

## Available Data Categories

- **Listing Information**: Stock symbols, exchanges, and industry classifications
- **Price Data**: Historical and intraday pricing, price depth, and trading data
- **Company Information**: Company profiles, officers, shareholders, and corporate events
- **Financial Data**: Income statements, balance sheets, cash flow statements, and financial ratios
- **Trading Data**: Real-time price boards and market data
- **Screening**: Stock screening capabilities based on various criteria

## Installation

1. Create a Python environment (recommended: Python 3.12+)
```bash
conda create -n stockqa python=3.12
conda activate stockqa
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key
   - Create a `.env` file with your API key: `OPENAI_API_KEY=your_api_key_here`
   - Or set it as an environment variable
   - Or enter it directly in the app when prompted

4. Run the application
```bash
streamlit run app.py
```

## Example Queries

- "Cho tôi thông tin về công ty VNM"
- "Giá cổ phiếu VIC trong 30 ngày qua"
- "So sánh giá cổ phiếu VNM, VIC và VHM"
- "Báo cáo tài chính của FPT quý gần nhất"
- "Liệt kê các công ty trong ngành ngân hàng"
- "Ai là cổ đông lớn nhất của VNM?"

## Credits

- [vnstock](https://github.com/thinh-vu/vnstock) - Vietnamese stock market data library
- [OpenAI](https://openai.com/) - For the AI language model and function calling capabilities
- [Streamlit](https://streamlit.io/) - For the web application framework 