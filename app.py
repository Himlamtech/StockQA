import os
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
from vnstock import Listing, Quote, Company, Finance, Trading, Screener

# Load environment variables from .env file
load_dotenv()

# Set page title and configuration
st.set_page_config(
    page_title="Stock Knowledge Q&A Bot", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

openai.api_key = openai_api_key

# Define function to handle function calling with OpenAI
def get_vnstock_function_call(query):
    """
    Use OpenAI's function calling to determine which vnstock function to use based on query.
    Returns the function name, parameters, and any extracted context.
    """
    functions = [
        {
            "type": "function",
            "function": {
                "name": "query_listing",
                "description": "Get information about stock listings, symbols, exchanges, and industries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": [
                                "all_symbols", "symbols_by_exchange", "symbols_by_group", 
                                "symbols_by_industries", "industries_icb"
                            ],
                            "description": "The specific function to call from Listing class"
                        },
                        "exchange": {
                            "type": "string",
                            "enum": ["HOSE", "HNX", "UPCOM"],
                            "description": "Stock exchange to query (if relevant)"
                        },
                        "group": {
                            "type": "string",
                            "enum": ["VN30", "VN100", "VNMID", "VNSML", "VNX50", "VNALL", "HNX30"],
                            "description": "Stock group to query (if relevant)" 
                        }
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_quote",
                "description": "Get stock price information and historical data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["history", "intraday", "price_depth"],
                            "description": "The specific function to call from Quote class"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to query"
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["1", "5", "15", "30", "60", "D", "W", "M"],
                            "description": "Time resolution for historical data"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["function_name", "symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_company",
                "description": "Get company information, profile, events, and news",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["profile", "officers", "shareholders", "subsidiaries", "overview", "events", "dividends", "news", "insider_deals"],
                            "description": "The specific function to call from Company class"
                        },
                        "symbol": {
                            "type": "string", 
                            "description": "Stock symbol to query"
                        }
                    },
                    "required": ["function_name", "symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_finance",
                "description": "Get financial information like income statements, balance sheets, and financial ratios",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["income_statement", "balance_sheet", "cash_flow", "ratio"],
                            "description": "The specific function to call from Finance class" 
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to query"
                        },
                        "period": {
                            "type": "string",
                            "enum": ["quarter", "year"],
                            "description": "Financial report period"
                        }
                    },
                    "required": ["function_name", "symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_trading",
                "description": "Get stock trading information and price board data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["price_board"],
                            "description": "The specific function to call from Trading class"
                        },
                        "symbols": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of stock symbols to query"
                        }
                    },
                    "required": ["function_name", "symbols"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "query_screener",
                "description": "Screen stocks based on various criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["stock"],
                            "description": "The specific function to call from Screener class"
                        }
                    },
                    "required": ["function_name"]
                }
            }
        }
    ]

    # Call OpenAI API with function calling
    response = openai.chat.completions.create(
        model="gpt-4o",  # Using latest model with function calling
        messages=[
            {"role": "system", "content": "You are a stock market analyst assistant that helps users get information about Vietnamese stocks using the vnstock library. Identify the appropriate function to call and parameters based on the user's query."},
            {"role": "user", "content": query}
        ],
        tools=functions,
        tool_choice="auto"
    )
    
    # Extract the function call
    message = response.choices[0].message
    if message.tool_calls:
        # Extract the function call
        function_call = message.tool_calls[0].function
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)
        
        # Return the extracted information
        return {
            "function": function_name,
            "arguments": arguments,
            "explanation": message.content
        }
    else:
        # No function call was made
        return {
            "function": None,
            "arguments": None,
            "explanation": message.content
        }

# Define functions to call appropriate vnstock functions
def execute_vnstock_function(function_info):
    """Execute the appropriate vnstock function based on the function call info"""
    function_type = function_info.get("function")
    arguments = function_info.get("arguments", {})
    
    if not function_type or not arguments:
        return None, "Could not determine the appropriate function to call."
    
    try:
        if function_type == "query_listing":
            return execute_listing_function(arguments)
        elif function_type == "query_quote":
            return execute_quote_function(arguments)
        elif function_type == "query_company":
            return execute_company_function(arguments)
        elif function_type == "query_finance":
            return execute_finance_function(arguments)
        elif function_type == "query_trading":
            return execute_trading_function(arguments)
        elif function_type == "query_screener":
            return execute_screener_function(arguments)
        else:
            return None, f"Unknown function type: {function_type}"
    except Exception as e:
        return None, f"Error executing function: {str(e)}"

def execute_listing_function(arguments):
    """Execute functions from the Listing class"""
    function_name = arguments.get("function_name")
    listing = Listing()
    
    if not hasattr(listing, function_name):
        return None, f"Function {function_name} not found in Listing class."
    
    func = getattr(listing, function_name)
    
    # Extract additional parameters
    kwargs = {k: v for k, v in arguments.items() if k != "function_name" and v is not None}
    
    # Call the function with parameters
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Error calling {function_name}: {str(e)}"

def execute_quote_function(arguments):
    """Execute functions from the Quote class"""
    function_name = arguments.get("function_name")
    symbol = arguments.get("symbol")
    
    if not symbol:
        return None, "Symbol is required for Quote functions."
    
    quote = Quote(symbol)
    
    if not hasattr(quote, function_name):
        return None, f"Function {function_name} not found in Quote class."
    
    func = getattr(quote, function_name)
    
    # Extract additional parameters
    kwargs = {k: v for k, v in arguments.items() if k not in ["function_name", "symbol"] and v is not None}
    
    # If dates are not provided, use reasonable defaults for history
    if function_name == "history" and "from_date" not in kwargs:
        # Default to last 30 days
        end_date = datetime.today()
        start_date = end_date - timedelta(days=30)
        kwargs["from_date"] = start_date.strftime("%Y-%m-%d")
        kwargs["to_date"] = end_date.strftime("%Y-%m-%d")
        kwargs["resolution"] = kwargs.get("resolution", "D")  # Default to daily
    
    # Call the function with parameters
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Error calling {function_name}: {str(e)}"

def execute_company_function(arguments):
    """Execute functions from the Company class"""
    function_name = arguments.get("function_name")
    symbol = arguments.get("symbol")
    
    if not symbol:
        return None, "Symbol is required for Company functions."
    
    company = Company(symbol)
    
    if not hasattr(company, function_name):
        return None, f"Function {function_name} not found in Company class."
    
    func = getattr(company, function_name)
    
    # Extract additional parameters
    kwargs = {k: v for k, v in arguments.items() if k not in ["function_name", "symbol"] and v is not None}
    
    # Call the function with parameters
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Error calling {function_name}: {str(e)}"

def execute_finance_function(arguments):
    """Execute functions from the Finance class"""
    function_name = arguments.get("function_name")
    symbol = arguments.get("symbol")
    period = arguments.get("period", "quarter")
    
    if not symbol:
        return None, "Symbol is required for Finance functions."
    
    finance = Finance(symbol=symbol, period=period)
    
    if not hasattr(finance, function_name):
        return None, f"Function {function_name} not found in Finance class."
    
    func = getattr(finance, function_name)
    
    # Extract additional parameters
    kwargs = {k: v for k, v in arguments.items() if k not in ["function_name", "symbol", "period"] and v is not None}
    
    # Call the function with parameters
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Error calling {function_name}: {str(e)}"

def execute_trading_function(arguments):
    """Execute functions from the Trading class"""
    function_name = arguments.get("function_name")
    symbols = arguments.get("symbols", [])
    
    if not symbols:
        return None, "Symbols list is required for Trading functions."
    
    trading = Trading()
    
    if not hasattr(trading, function_name):
        return None, f"Function {function_name} not found in Trading class."
    
    func = getattr(trading, function_name)
    
    # Extract additional parameters
    kwargs = {k: v for k, v in arguments.items() if k not in ["function_name", "symbols"] and v is not None}
    
    # Call the function with parameters
    try:
        result = func(symbols_list=symbols, **kwargs)
        return result, None
    except Exception as e:
        return None, f"Error calling {function_name}: {str(e)}"

def execute_screener_function(arguments):
    """Execute functions from the Screener class"""
    function_name = arguments.get("function_name")
    
    screener = Screener()
    
    if not hasattr(screener, function_name):
        return None, f"Function {function_name} not found in Screener class."
    
    func = getattr(screener, function_name)
    
    # Extract additional parameters
    kwargs = {k: v for k, v in arguments.items() if k != "function_name" and v is not None}
    
    # Call the function with parameters
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Error calling {function_name}: {str(e)}"

def process_data_for_display(data, function_name):
    """Process the data returned by vnstock for better display in Streamlit"""
    if data is None:
        return None
    
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        else:
            try:
                data = pd.DataFrame(data)
            except:
                return data  # Return as is if we can't convert
    
    # Apply specific processing based on the function type
    if function_name == "query_quote" and data.shape[0] > 0:
        # For historical price data, we might want to create a chart
        if 'time' in data.columns and 'close' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['time'], data['close'])
            ax.set_title(f"Stock Price History")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            return {"data": data, "chart": fig}
    
    # For company profile, we might want to format it nicely
    if function_name == "query_company":
        # Special formatting for company profile
        pass
    
    # For financial statements, we might want to highlight important metrics
    if function_name == "query_finance":
        # Special formatting for financial data
        pass
    
    return data

def generate_response(query, data, explanation):
    """
    Generate a comprehensive response based on the query and data.
    Uses OpenAI to create a natural language answer.
    """
    # Convert data to a string representation for the OpenAI API
    if isinstance(data, pd.DataFrame):
        data_str = data.to_string(index=True)
    elif isinstance(data, dict) and "data" in data and "chart" in data:
        # Handle the case with DataFrame and chart
        data_str = data["data"].to_string(index=True)
        # We don't include the chart in the string, as it's visual
    else:
        data_str = str(data)
    
    # Limit data string size if needed
    if len(data_str) > 8000:  # Truncate if too large
        data_str = data_str[:8000] + "... [truncated]"
    
    # Call OpenAI API to generate a natural language response
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial analyst expert specializing in Vietnamese stock market. Provide clear, accurate, and insightful analysis based on the data provided. Use Vietnamese language in your response since this is for Vietnamese users."},
            {"role": "user", "content": f"Query: {query}\n\nData: {data_str}\n\nBased on this data, provide a comprehensive analysis and answer to the query. Explain key insights, trends, and implications in a way that's easy to understand."}
        ]
    )
    
    return response.choices[0].message.content

# Streamlit UI
st.title("🇻🇳 Vietnam Stock Market Q&A Bot")
st.markdown("Hỏi đáp thông tin về chứng khoán Việt Nam")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "data" in message:
            # Display data if present (for user queries)
            if isinstance(message["data"], pd.DataFrame):
                st.dataframe(message["data"])
            elif isinstance(message["data"], dict) and "chart" in message["data"]:
                # Display both chart and data
                st.pyplot(message["data"]["chart"])
                with st.expander("Show Data"):
                    st.dataframe(message["data"]["data"])
        # Display the message content
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Hỏi thông tin về chứng khoán...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant "thinking" message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("⏳ Đang tìm kiếm thông tin...")
        
        # Process the query with OpenAI function calling
        function_info = get_vnstock_function_call(query)
        
        # Log the function info for debugging
        st.session_state.function_info = function_info
        
        # Execute the appropriate vnstock function
        if function_info.get("function"):
            thinking_placeholder.markdown("⏳ Đang truy xuất dữ liệu từ vnstock...")
            data, error = execute_vnstock_function(function_info)
            
            if error:
                thinking_placeholder.markdown(f"❌ Lỗi: {error}")
                response_content = f"Tôi xin lỗi, tôi gặp vấn đề khi truy xuất dữ liệu: {error}. Vui lòng thử lại với câu hỏi khác hoặc điều chỉnh yêu cầu của bạn."
            else:
                # Process the data for better display
                processed_data = process_data_for_display(data, function_info.get("function"))
                
                # Generate natural language response
                thinking_placeholder.markdown("⏳ Đang phân tích dữ liệu và tạo câu trả lời...")
                response_content = generate_response(query, processed_data, function_info.get("explanation"))
                
                # Display the data
                if processed_data is not None:
                    if isinstance(processed_data, pd.DataFrame):
                        st.dataframe(processed_data)
                    elif isinstance(processed_data, dict) and "chart" in processed_data:
                        st.pyplot(processed_data["chart"])
                        with st.expander("Show Data"):
                            st.dataframe(processed_data["data"])
        else:
            # Handle case where no function was called
            response_content = function_info.get("explanation", "Tôi không thể xác định chính xác thông tin bạn cần. Vui lòng cung cấp thêm chi tiết về mã chứng khoán hoặc loại thông tin bạn đang tìm kiếm.")
        
        # Clear the thinking message and display the final response
        thinking_placeholder.empty()
        st.markdown(response_content)
    
    # Add assistant response to chat history
    assistant_message = {
        "role": "assistant",
        "content": response_content
    }
    
    # Add the data to the message if available
    if function_info.get("function") and 'data' in locals() and data is not None:
        processed_data = locals().get('processed_data')
        if processed_data is not None:
            assistant_message["data"] = processed_data
    
    st.session_state.messages.append(assistant_message)

# Add a sidebar with information about the app
with st.sidebar:
    st.header("Về ứng dụng")
    st.markdown("""
    **Hỏi đáp thông tin về Chứng khoán Việt Nam** là ứng dụng giúp bạn tìm kiếm và phân tích thông tin về thị trường chứng khoán Việt Nam.
    
    Bạn có thể hỏi về:
    - Thông tin công ty và mã chứng khoán
    - Giá cổ phiếu và dữ liệu giao dịch
    - Thông tin tài chính và báo cáo
    - Phân tích và so sánh cổ phiếu
    
    Dữ liệu được cung cấp bởi thư viện [vnstock](https://github.com/thinh-vu/vnstock).
    """)
    
    st.header("Ví dụ câu hỏi")
    st.markdown("""
    - "Cho tôi thông tin về công ty VNM"
    - "Giá cổ phiếu VIC trong 30 ngày qua"
    - "So sánh giá cổ phiếu VNM, VIC và VHM"
    - "Báo cáo tài chính của FPT quý gần nhất"
    - "Liệt kê các công ty trong ngành ngân hàng"
    - "Ai là cổ đông lớn nhất của VNM?"
    """)

# Run the app with: streamlit run app.py 