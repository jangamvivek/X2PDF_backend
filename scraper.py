import os
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
from typing import Dict, Any, List
import io
import re
import logging
from config import GOOGLE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using Google's Gemini API
    """
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load the image
        image = Image.open(image_path)
        logger.info(f"Image loaded successfully. Size: {image.size}, Mode: {image.mode}")
        
        # Initialize Gemini 1.5 Flash model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content from image with specific instructions
        response = model.generate_content([
            """Extract all text from this image and format it as structured data.
            If you see tables, lists, or structured information, maintain that structure.
            For tables, use column headers as keys and rows as values.
            For lists, use numbered or bulleted format.
            For key-value pairs, use the format 'key: value'.
            Try to identify and maintain any numerical data, dates, or special formatting.""",
            image
        ])
        
        extracted_text = response.text
        logger.info(f"Text extracted successfully. Length: {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from image: {str(e)}")

def clean_text(text: str) -> str:
    """
    Clean and normalize the extracted text
    """
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,:;()\-+*/=<>$%]', '', text)
        cleaned_text = text.strip()
        logger.info(f"Text cleaned successfully. Length: {len(cleaned_text)}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}", exc_info=True)
        raise Exception(f"Failed to clean text: {str(e)}")

def parse_structured_data(text: str) -> Dict[str, Any]:
    """
    Parse the extracted text into structured data
    """
    try:
        # Clean the text first
        text = clean_text(text)
        
        # Try to parse as JSON first
        try:
            data = json.loads(text)
            logger.info("Successfully parsed text as JSON")
            return data
        except json.JSONDecodeError:
            logger.info("Text is not in JSON format, attempting to parse as structured text")
            pass
            
        # Try to identify table structure
        lines = text.strip().split('\n')
        data = {}
        current_section = None
        table_data = []
        headers = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for table headers
            if '|' in line:
                if headers is None:
                    headers = [h.strip() for h in line.split('|') if h.strip()]
                    continue
                else:
                    # Process table row
                    values = [v.strip() for v in line.split('|') if v.strip()]
                    if len(values) == len(headers):
                        row = dict(zip(headers, values))
                        table_data.append(row)
                    continue
            
            # Check for key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                try:
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r'^-?\d*\.\d+$', value):
                        value = float(value)
                except ValueError:
                    pass
                    
                data[key] = value
                continue
            
            # Check for numbered lists
            if re.match(r'^\d+\.', line):
                if current_section is None:
                    current_section = 'numbered_list'
                    data[current_section] = []
                data[current_section].append(line)
                continue
            
            # Check for bullet points
            if line.startswith('•') or line.startswith('-'):
                if current_section is None:
                    current_section = 'bullet_list'
                    data[current_section] = []
                data[current_section].append(line.lstrip('•- '))
                continue
            
            # Regular text
            if current_section is None:
                current_section = 'text'
                data[current_section] = []
            data[current_section].append(line)
        
        # Add table data if found
        if table_data:
            data['table'] = table_data
            
        logger.info(f"Successfully parsed structured data with {len(data)} sections")
        return data
    except Exception as e:
        logger.error(f"Error parsing structured data: {str(e)}", exc_info=True)
        raise Exception(f"Failed to parse structured data: {str(e)}")

def convert_to_excel(data: Dict[str, Any], output_path: str) -> str:
    """
    Convert the structured data to Excel format
    """
    try:
        logger.info(f"Converting data to Excel: {output_path}")
        
        # Create a list of DataFrames for different sections
        dfs = []
        
        # Process table data if present
        if 'table' in data:
            df_table = pd.DataFrame(data['table'])
            dfs.append(('Table Data', df_table))
        
        # Process numbered list if present
        if 'numbered_list' in data:
            df_numbered = pd.DataFrame(data['numbered_list'], columns=['Items'])
            dfs.append(('Numbered List', df_numbered))
        
        # Process bullet list if present
        if 'bullet_list' in data:
            df_bullet = pd.DataFrame(data['bullet_list'], columns=['Items'])
            dfs.append(('Bullet List', df_bullet))
        
        # Process key-value pairs
        kv_data = {k: v for k, v in data.items() if k not in ['table', 'numbered_list', 'bullet_list', 'text']}
        if kv_data:
            df_kv = pd.DataFrame([kv_data])
            dfs.append(('Key-Value Pairs', df_kv))
        
        # Process text if present
        if 'text' in data:
            df_text = pd.DataFrame(data['text'], columns=['Text'])
            dfs.append(('Text Content', df_text))
        
        # If we have table data, use it as the main sheet
        if 'table' in data:
            main_df = pd.DataFrame(data['table'])
        # Otherwise, combine all data into a single DataFrame
        else:
            main_df = pd.DataFrame()
            for _, df in dfs:
                if not main_df.empty:
                    main_df = pd.concat([main_df, df], axis=1)
                else:
                    main_df = df
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save main DataFrame as the first sheet
            main_df.to_excel(writer, sheet_name='Main Data', index=False)
            # Save other DataFrames as additional sheets
            for sheet_name, df in dfs:
                if df is not main_df:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Successfully saved Excel file: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error converting to Excel: {str(e)}", exc_info=True)
        raise Exception(f"Failed to convert data to Excel: {str(e)}")

def convert_to_csv(data: Dict[str, Any], output_path: str) -> str:
    """
    Convert the structured data to CSV format
    """
    try:
        logger.info(f"Converting data to CSV: {output_path}")
        
        # If we have table data, use it as the main data
        if 'table' in data:
            df = pd.DataFrame(data['table'])
        else:
            # Combine all data into a single DataFrame
            flat_data = []
            
            # Process numbered list if present
            if 'numbered_list' in data:
                flat_data.extend([{'type': 'numbered_list', 'content': item} for item in data['numbered_list']])
            
            # Process bullet list if present
            if 'bullet_list' in data:
                flat_data.extend([{'type': 'bullet_list', 'content': item} for item in data['bullet_list']])
            
            # Process key-value pairs
            kv_data = {k: v for k, v in data.items() if k not in ['table', 'numbered_list', 'bullet_list', 'text']}
            if kv_data:
                flat_data.append(kv_data)
            
            # Process text if present
            if 'text' in data:
                flat_data.extend([{'type': 'text', 'content': item} for item in data['text']])
            
            df = pd.DataFrame(flat_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved CSV file: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error converting to CSV: {str(e)}", exc_info=True)
        raise Exception(f"Failed to convert data to CSV: {str(e)}")

def process_image(image_path: str, output_format: str = 'excel') -> str:
    """
    Process an image to extract text and convert to Excel/CSV
    """
    try:
        logger.info(f"Starting image processing: {image_path}")
        
        # Extract text from image
        extracted_text = extract_text_from_image(image_path)
        if not extracted_text:
            raise Exception("No text was extracted from the image")
            
        # Parse the extracted text into structured data
        structured_data = parse_structured_data(extracted_text)
        if not structured_data:
            raise Exception("Failed to parse extracted text into structured data")
            
        # Generate output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        
        if output_format.lower() == 'excel':
            output_path = os.path.join(output_dir, f"{base_name}.xlsx")
            return convert_to_excel(structured_data, output_path)
        else:
            output_path = os.path.join(output_dir, f"{base_name}.csv")
            return convert_to_csv(structured_data, output_path)
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process image: {str(e)}") 