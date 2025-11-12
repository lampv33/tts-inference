import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types
from speedy_utils import memoize
import json
import re
import torch
from collections import defaultdict
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


client = genai.Client(api_key='AIzaSyDRTq_J0wnDlFph5N52rbioUNJcF-AEeWw')


class StyleSManager:
    """
    Quản lý segment audio từ file Excel với bộ lọc kép: độ dài audio và
    số lượng tối thiểu segment cho mỗi style.
    """
    def __init__(self,
                 style_file_path: str,
                 min_duration_seconds: float = 5.0,
                 min_segments_per_style: int = 1):
        """
        Khởi tạo manager bằng cách đọc và xử lý file Excel.

        Args:
            style_file_path (str): Đường dẫn đến file Excel.
            min_duration_seconds (float): Độ dài tối thiểu của một segment để được giữ lại.
            min_segments_per_style (int): Số lượng segment tối thiểu mà một style
                                          phải có để không bị loại bỏ.
        """
        if not os.path.exists(style_file_path):
            raise FileNotFoundError(f"Không tìm thấy file tại: {style_file_path}")
            
        self.excel_file_path = style_file_path
        self.min_duration_seconds = min_duration_seconds
        self.min_segments_per_style = min_segments_per_style
        self._all_segments = []
        self._segments_by_style = defaultdict(list)
        
        self._load_data()

    def _load_data(self):
        """Đọc file Excel và thực hiện lọc 2 bước."""
        print(f"--- Bắt đầu quá trình tải và lọc dữ liệu từ '{os.path.basename(self.excel_file_path)}' ---")
        df = pd.read_excel(self.excel_file_path)
        
        if 'segment_path' not in df.columns or 'style' not in df.columns:
            raise ValueError("File Excel phải chứa các cột 'segment_path' và 'style'.")
        
        # --- BƯỚC 1: LỌC THEO ĐỘ DÀI AUDIO ---
        print(f"Bước 1: Lọc segment (giữ lại những segment >= {self.min_duration_seconds} giây)...")
        
        temp_segments_by_style = defaultdict(list)
        initial_count = len(df)
        skipped_duration = 0
        skipped_error = 0

        for _, row in df.iterrows():
            segment_path, style = row.get('segment_path'), row.get('style')
            if pd.isna(style) or pd.isna(segment_path): continue
            
            try:
                audio = AudioSegment.from_file(segment_path)
                if audio.duration_seconds < self.min_duration_seconds:
                    skipped_duration += 1
                    continue
            except (FileNotFoundError, CouldntDecodeError):
                skipped_error += 1
                continue
            
            temp_segments_by_style[style].append(segment_path)

        print(f" -> Lọc theo độ dài hoàn tất. Giữ lại {initial_count - skipped_duration - skipped_error}/{initial_count} segment.")
        
        # --- BƯỚC 2: LỌC THEO SỐ LƯỢNG SEGMENT MỖI STYLE ---
        print(f"\nBước 2: Lọc style (giữ lại những style có >= {self.min_segments_per_style} segment)...")
        
        initial_style_count = len(temp_segments_by_style)
        final_segment_count = 0
        
        for style, segments in temp_segments_by_style.items():
            if len(segments) >= self.min_segments_per_style:
                self._segments_by_style[style] = segments
                self._all_segments.extend([{"segment_path": p, "style": style} for p in segments])
                final_segment_count += len(segments)
        
        final_style_count = len(self._segments_by_style)
        print(f" -> Lọc theo số lượng hoàn tất. Giữ lại {final_style_count}/{initial_style_count} style.")
        
        print("\n--- Tải dữ liệu hoàn tất! ---")
        print(f"   - Tổng số segment cuối cùng: {final_segment_count}")
        print(f"   - Tổng số style cuối cùng: {final_style_count}")
        print(f"   - Bỏ qua do quá ngắn: {skipped_duration} segment.")
        print(f"   - Bỏ qua do lỗi file: {skipped_error} segment.")
        print("-" * 40)
        
    def get_all_segments(self) -> list:
        return self._all_segments

    def get_segment_by_style_and_id(self, style: str, segment_id: int) -> str | None:
        if style not in self._segments_by_style:
            print(f"Cảnh báo: Không tìm thấy style '{style}' trong dữ liệu đã lọc.")
            return None
            
        segments_for_style = self._segments_by_style[style]
        index = segment_id - 1
        
        if not (0 <= index < len(segments_for_style)):
            print(f"Cảnh báo: ID '{segment_id}' không hợp lệ cho style '{style}'.")
            print(f"         Style này chỉ có {len(segments_for_style)} segment (ID từ 1 đến {len(segments_for_style)}).")
            return None
            
        return segments_for_style[index]
    

@memoize
def split_text_by_style(text, styles_list):
    prompt = f'''Bạn được cho một câu chuyện và list các style giọng. Hãy chia câu chuyện thành các đoạn nhỏ theo thứ tự và chọn một style giọng phù hợp cho mỗi đoạn.
    Câu chuyện: {text}
    List style giọng: {styles_list}
    
    Yêu cầu:
    - Chỉ được chọn style giọng trong list
    - Chia đoạn phù hợp, sao cho mỗi đoạn có thể đọc bởi cùng 1 style/tone giọng
    - Chia ít đoạn/style nhất có thể để đảm bảo tính nhất quán và không chuyển style quá nhiều lần, đặc biệt là giọng dẫn chuyện. Chỉ tách đoạn (đổi style) khi cần thiết
    - Trả về kết quả theo format json. Sample output:
        
        [
            {{
                "paragraph": "Tấm tin là thật, bèn xuống ao lặn một lúc lâu. Trong khi đó, Cám ở trên bờ đã trút hết giỏ tép của Tấm vào giỏ mình rồi ba chân bốn cẳng chạy về nhà trước.",
                "style": "tường thuật"
            }},
            {{
                "paragraph": "Khi Tấm bước lên, thấy giỏ tép trống không, cô chỉ biết ngồi bên bờ ruộng, tủi thân òa khóc nức nở.",
                "style": "đau buồn"
            }}
        ]
    '''

    response = client.models.generate_content(
        model='gemini-2.5-pro',     
        config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=128)),
        contents=prompt
    )

    return response.text


def extract_and_parse_json(text_response: str) -> list | dict | None:
    """
    Trích xuất một chuỗi JSON từ phản hồi văn bản (thường từ LLM) và phân tích nó.

    Hàm này xử lý các định dạng phổ biến của LLM, bao gồm:
    - Chuỗi JSON thuần túy.
    - JSON được bọc trong khối mã Markdown (```json ... ```).
    - JSON bị lẫn trong các đoạn văn bản khác.

    Args:
        text_response: Chuỗi văn bản thô từ LLM.

    Returns:
        Một list hoặc dict của Python nếu phân tích thành công, ngược lại trả về None.
    """
    # Phương pháp 1: Tìm và phân tích khối mã Markdown ```json
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text_response)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Lỗi: Nội dung bên trong khối JSON bị lỗi cú pháp. Lỗi: {e}")
            print("--- Nội dung lỗi ---")
            print(json_str)
            print("-------------------")
            return None

    # Phương pháp 2: Nếu không có khối markdown, thử tìm từ dấu '[' hoặc '{' đầu tiên
    # Đây là cách "cứu cánh" khá hiệu quả khi JSON bị lẫn với văn bản giới thiệu.
    start_index = -1
    first_bracket = text_response.find('[')
    first_curly = text_response.find('{')

    if first_bracket != -1 and first_curly != -1:
        start_index = min(first_bracket, first_curly)
    elif first_bracket != -1:
        start_index = first_bracket
    else:
        start_index = first_curly

    if start_index != -1:
        json_candidate = text_response[start_index:]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            # Lỗi có thể do có văn bản thừa ở cuối, tiếp tục thử
            pass
            
    # Phương pháp 3: Nếu tất cả đều thất bại, báo lỗi và trả về None
    print("Lỗi: Không thể phân tích chuỗi phản hồi thành JSON.")
    print("--- Phản hồi gốc ---")
    print(text_response)
    print("--------------------")
    return None



def process_and_save_style_embeddings(excel_file_path, model, top_n=None):
    """
    Đọc file Excel, tính toán embedding trung bình cho top N style có nhiều segment nhất,
    và lưu kết quả vào một thư mục được tạo tự động.

    Thư mục đầu ra sẽ có cấu trúc: 'ref_embeddings/{model.model_name}/{excel_basename}/'
    và sẽ bị xóa nếu đã tồn tại trước khi chạy.

    Args:
        excel_file_path (str):
            Đường dẫn đến file Excel. Phải chứa các cột 'segment_path' và 'style'.

        model (object):
            Đối tượng model đã được khởi tạo. PHẢI có các thuộc tính:
            - `model_name` (str): Tên của model.
            - `compute_ref_emb(path)` (method): Phương thức trả về một numpy array.

        top_n (int, optional):
            Chỉ xử lý top N style có nhiều segment nhất.
            Nếu là None, tất cả các style sẽ được xử lý. Mặc định là None.
    """
    # --- 1. Xây dựng đường dẫn đầu ra, đọc và lọc dữ liệu ---
    base_name = os.path.splitext(os.path.basename(excel_file_path))[0]
    
    try:
        model_name = model.model_name
    except AttributeError:
        print("Lỗi: Đối tượng 'model' được cung cấp phải có thuộc tính 'model_name'.")
        return

    output_dir = os.path.join('ref_embeddings', model_name, base_name)

    # Xóa thư mục đầu ra nếu nó đã tồn tại
    if os.path.exists(output_dir):
        #print(f"Thư mục '{output_dir}' đã tồn tại. Đang xóa...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{excel_file_path}'. Dừng chương trình.")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file Excel: {e}")
        return

    df.dropna(subset=['style'], inplace=True)
    if df.empty:
        print("Cảnh báo: File Excel không chứa dữ liệu style hợp lệ.")
        return

    # Logic xử lý top_n
    if top_n and isinstance(top_n, int) and top_n > 0:
        #print(f"\nChỉ xử lý top {top_n} style có nhiều segment nhất...")
        # Đếm số lượng segment cho mỗi style
        style_counts = df['style'].value_counts()
        
        # Lấy top N style và số lượng segment tương ứng
        top_n_style_counts = style_counts.nlargest(top_n)
        top_styles = top_n_style_counts.index.tolist()
        
        # Lọc DataFrame để chỉ giữ lại các style trong top N
        df = df[df['style'].isin(top_styles)]
        
        # In ra thông tin
        #print(top_styles)
        if not top_n_style_counts.empty:
            min_segments_in_top_n = top_n_style_counts.min()
            print(f"Các style trong top {top_n} có ít nhất {min_segments_in_top_n} audio.\n")

    else:
        print("\nXử lý tất cả các style được tìm thấy...")
        
    initial_unique_styles = sorted(list(df['style'].unique()))
    
    # --- 2. Tính toán và lưu embedding ---
    grouped_by_style = df.groupby('style')
    successfully_processed_styles = []
    
    for style_name, group in tqdm(grouped_by_style, total=len(initial_unique_styles), desc=f"Processing"):
        embeddings_for_style = []
        
        for segment_path in group['segment_path']:
            try:
                embedding = model.compute_ref_emb(segment_path, display_audio=False).cpu().numpy()
                embeddings_for_style.append(embedding)
            except Exception:
                continue
            
        if not embeddings_for_style:
            tqdm.write(f"Cảnh báo: Không có segment audio hợp lệ nào cho style '{style_name}'. Bỏ qua style này.")
            continue

        avg_emb = np.mean(embeddings_for_style, axis=0)
        output_npy_filename = f"{style_name}.npy"
        output_npy_path = os.path.join(output_dir, output_npy_filename)
        np.save(output_npy_path, avg_emb)
        successfully_processed_styles.append(style_name)

    # --- 3. Lưu danh sách các style đã xử lý thành công ---
    if successfully_processed_styles:
        styles_txt_path = os.path.join(output_dir, 'styles.txt')
        try:
            successfully_processed_styles.sort()
            with open(styles_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(successfully_processed_styles) + '\n')
        except IOError as e:
            print(f"Lỗi: Không thể ghi file styles.txt. Lỗi: {e}")

    # --- 4. Báo cáo cuối cùng ---
    num_success = len(successfully_processed_styles)
    num_total = len(initial_unique_styles)
    
    print("\n" + "="*50)
    print(f"Đã xử lý và lưu thành công kết quả cho {num_success}/{num_total} style.")
    #print(f"Dữ liệu được lưu tại: '{output_dir}'.")
    #print("="*50)
    print(successfully_processed_styles)
    print()

    return successfully_processed_styles, output_dir


def get_embedding_maiyen(model):
    embedding = model.compute_ref_emb('/mnt/nfs-shared/kilm/users/lampv/prepare_data_tts/segments/leyen-voizfm/EnVOmNgen6k_36.wav', display_audio=False)
    return embedding




def get_embedding_story_2(ref_embedding, embedding_maiyen):
    embedding_voice = embedding_maiyen[:, :128]
    #embedding_voice_ref = ref_embedding[:, :128]
    embedding_style = ref_embedding[:, 128:]
    #embedding_voice = (embedding_voice_maiyen * 2 + embedding_voice_ref) / 3
    embedding = torch.cat([embedding_voice, embedding_style], dim=-1)

    return embedding

    
