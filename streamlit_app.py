import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import numpy as np
from typing import List, Dict, Optional, Union
import plotly.express as px

# Import file utils của bạn
try:
    import utils_optimized as utils 
except ImportError:
    st.error("LỖI: Không tìm thấy file utils_optimized.py! Hãy kiểm tra file và packages.txt")
    st.stop()

# =================================================================
# I. KHỐI CẤU HÌNH & CSS (CHẠY ĐẦU TIÊN)
# =================================================================
# --- TRONG streamlit_app.py (Ngay sau các khối import) ---

# Đảm bảo bạn đã đặt cấu hình này CHÍNH XÁC:
st.set_page_config(
    page_title="Báo cáo Tăng trưởng Thị trường",
    page_icon="📈", # <-- ICON ĐƯỢC CHỌN
    layout="wide"
)

# ... (Khối CSS CUSTOM_CSS của bạn bắt đầu tại đây) ...

# --- 1. CSS STYLING (CHỈ DÁN 1 LẦN) ---
# --- 1. CSS STYLING (ĐÃ DỌN DẸP & TỐI ƯU HÓA) ---
CUSTOM_CSS = """
/* Thiết lập Font chính */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st"] {
    font-family: 'Inter', sans-serif;
}

/* Nâng cấp containers/block */
div[data-testid="stVerticalBlock"], div[data-testid="stExpander"] {
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: box-shadow 0.3s ease-in-out;
}

/* Nâng cấp cho Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: nowrap;
    background-color: #f0f2f6;
    border-radius: 8px 8px 0 0;
    padding: 0px 15px;
    margin-right: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.stTabs [aria-selected="true"] {
    border-bottom: 4px solid #F26522; /* Orange Line */
    font-weight: bold;
    color: #F26522;
    background-color: white;
    box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.1);
}

/* KPI Card Custom Styling */
.metric-card { 
    background-color: var(--st-secondary-background);
    padding: 15px; 
    border-radius: 10px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
    transition: transform 0.2s ease;
    margin-bottom: 10px;
    height: 120px;
}
.metric-card:hover {
    box-shadow: 0 6px 12px rgba(0,0,0,0.15); 
}

/* Style cho Banner */
.banner { 
    background: linear-gradient(90deg, #005566, #F28C38); 
    color: white; 
    padding: 30px; 
    text-align: center; 
    border-radius: 12px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin-bottom: 20px;
}

/* Thêm style cho DataFrame */
.stDataFrame, .stDataEditor {
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}


/* Cần nâng cấp Streamlit để lỗi này tự biến mất. */
/* Ẩn nút mặc định bị lỗi */
.st-emotion-cache-1f87s41, .st-emotion-cache-1s0l76, .st-emotion-cache-1p07vfl { 
    visibility: hidden !important; 
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    position: absolute;
    left: -100px; /* Đẩy hẳn ra khỏi màn hình */
}
/* Ẩn vùng chứa lỗi text */
.st-emotion-cache-1629p8f button p {
    display: none !important;
}

"""
st.markdown(f'<style>{CUSTOM_CSS}</style>', unsafe_allow_html=True)


# --- 2. HÀM KPI CARD (NÓI CHUYỆN VỚI CSS) ---
def style_kpi_card(title, value, delta_value, color_pos='#03943F', color_neg='#B22F16'):
    """
    Creates a styled metric card using markdown and custom CSS.
    """
    try:
        if isinstance(delta_value, str):
            delta = float(delta_value.strip('%'))
        else:
            delta = float(delta_value)
    except (ValueError, AttributeError):
        delta = 0
        delta_value = "N/A"

    color = color_pos if delta > 0 else color_neg if delta < 0 else '#636466'
    sign = '▲' if delta > 0 else '▼' if delta < 0 else '—'
    
    html = f"""
    <div class='metric-card' style='border-left: 5px solid {color};'>
        <p style='font-size: 14px; color: #636466; margin-bottom: 5px;'>{title}</p>
        <h3 style='font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 5px;'>{value}</h3>
        <p style='font-size: 16px; color: #333333; font-weight: 600;'>
            {sign} {abs(delta):.2f}%
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --- 3. HÀM STYLE TABLE HEADER (NÓI CHUYỆN VỚI PANDAS) ---
def style_table_header(df):
    """Áp dụng styling cho header của DataFrame (Xanh dương #1569B4)"""
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#1569B4'), 
            ('color', 'white'), 
            ('font-weight', '600'),
            ('border-radius', '8px 8px 0 0'),
            ('text-align', 'center')
        ]}
    ])

# =================================================================
# II. TẢI VÀ CHUẨN BỊ DỮ LIỆU (ĐỘC LẬP)
# =================================================================
DATA_DIR = "data_cache" 

def get_current_period():
    today = datetime.date.today()
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1
    if current_quarter == 1:
        return current_year - 1, "Q4"
    else:
        return current_year, f"Q{current_quarter - 1}"

DEFAULT_YEAR, DEFAULT_QUARTER = get_current_period()
print(f"Kỳ mặc định được chọn: {DEFAULT_QUARTER} {DEFAULT_YEAR}")

@st.cache_data(ttl=600)
def load_data_from_cache():
    print(f"[{datetime.datetime.now()}] Đang tải dữ liệu từ cache (Hybrid)...")
    try:
        merged_file = os.path.join(DATA_DIR, "df_merged.parquet")
        stats_file = os.path.join(DATA_DIR, "df_market_stats_historical.parquet")
        
        if not os.path.exists(merged_file) or not os.path.exists(stats_file):
            st.error(f"Lỗi: Không tìm thấy file cache trong '{DATA_DIR}'.")
            st.info("Hãy đảm bảo rằng tác vụ chạy nền (run_batch_job.py) đã chạy thành công ít nhất một lần.")
            return None, None, None, None

        last_updated_time = os.path.getmtime(merged_file)
        last_updated_str = datetime.datetime.fromtimestamp(last_updated_time).strftime('%Y-%m-%d %H:%M:%S')

        df_merged = pd.read_parquet(merged_file)
        df_market_stats_historical = pd.read_parquet(stats_file)
        
        print("Tải cache (Hybrid) thành công!")
        return last_updated_str, df_merged, df_market_stats_historical
    
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi đọc file cache: {e}")
        return None, None, None, None

# Tải dữ liệu chính
data = load_data_from_cache()
last_updated, df_merged, df_market_stats_historical = data

# Kiểm tra lỗi tải dữ liệu
if df_merged is None:
    st.stop()


# =================================================================
# III. BỘ LỌC VÀ BANNER (THỨ TỰ CHÍNH XÁC)
# =================================================================

# --- 1. BỘ LỌC SIDEBAR ---
st.sidebar.markdown('<p style="font-size: 1.5em; font-weight: bold; color: #034EA2;">⚙️ Bộ lọc Báo cáo</p>', unsafe_allow_html=True)

# Lấy danh sách Năm và Quý từ dữ liệu đã có
available_years = sorted(df_merged['Nam'].unique(), reverse=True)
available_quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# Tìm index của giá trị mặc định
try: year_index = available_years.index(DEFAULT_YEAR)
except ValueError: year_index = 0

try: quarter_index = available_quarters.index(DEFAULT_QUARTER)
except ValueError: quarter_index = 0

select_year = st.sidebar.selectbox(
    "Chọn Năm xem Báo cáo:",
    available_years,
    index=year_index
)
select_quarter = st.sidebar.selectbox(
    "Chọn Quý xem Báo cáo:",
    available_quarters,
    index=quarter_index
)

st.sidebar.success(f"Đang hiển thị báo cáo cho: **{select_quarter} {select_year}**")


# --- 2. BANNER CHÍNH (SỬ DỤNG BIẾN BỘ LỌC) ---
st.markdown(f"""
    <div class='banner'>
        <h1 style='font-size: 28px; margin: 0; padding: 0;'>
            BÁO CÁO KẾT QUẢ TÀI CHÍNH
            <br>
            <span style='font-size: 18px; font-weight: 400;'>
            QUÝ {select_quarter[-1]} NĂM {select_year}
            </span>
        </h1>
    </div>
""", unsafe_allow_html=True)




# Giữ lại thời gian cập nhật
if last_updated:
    st.caption(f"Dữ liệu gốc được cập nhật lần cuối lúc: **{last_updated}** (Giờ máy chủ)")

# --- DÁN KHỐI HTML/JS NÀY VÀO SAU st.caption(...) ---


# =================================================================
# IV. LOGIC TÍNH TOÁN ON-THE-FLY VÀ TẠO TAB
# =================================================================

@st.cache_data(ttl=600)
def calculate_report_for_period(_df_merged, year, quarter):
    print(f"[{datetime.datetime.now()}] Đang tính toán on-the-fly cho {year}-{quarter}...")
    
    # 1. Tính tăng trưởng ngành (Sử dụng Phân ngành - ICB L2)
    df_industry = utils.calculate_industry_growth_rates_abs_base(
        _df_merged,
        industry_col='Phân ngành - ICB L2',
        filter_year=year,
        filter_quarter=int(quarter[1:])
    )
    
    # 2. Phân tích Top 20 (Sử dụng hàm từ utils_optimized.py)
    top_results = utils.analyze_top_10_stocks(
        _df_merged,
        current_year=year,
        current_quarter=quarter,
        top_n=20
    )
    
    # 3. Lấy tóm tắt thị trường (chỉ phần tăng trưởng)
    if df_industry is not None and not df_industry.empty:
        df_market_summary = df_industry[df_industry['Phân loại'] == 'Toàn thị trường']
    else:
        df_market_summary = pd.DataFrame()
    
    print(f"   ...Tính toán on-the-fly cho {year}-{quarter} HOÀN TẤT.")
    return df_industry, top_results, df_market_summary

# Thực hiện tính toán On-the-fly (Dùng kết quả của bộ lọc)
df_industry, top_results, df_market_summary = calculate_report_for_period(
    df_merged, select_year, select_quarter
)

if df_industry is None or df_industry.empty:
    st.error("Không có đủ dữ liệu để tạo báo cáo chi tiết cho kỳ này. Vui lòng chọn kỳ khác.")
    st.stop()


# --- KHỞI TẠO TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Thị trường", "🔍 Chi tiết Ngành", "🥇 Top Cổ phiếu"])

# =================================================================
# V. NỘI DUNG TỪNG TAB
# =================================================================

# -----------------------------------------------
# TAB 1: TỔNG QUAN THỊ TRƯỜNG
# -----------------------------------------------
with tab1:
    
    # 1. BẢNG TĂNG TRƯỞNG CHÍNH (6 KPIs)
    st.markdown("<h3 style='color: #1569B4;'>🎯 Tăng trưởng Toàn thị trường</h3>", unsafe_allow_html=True)

    # Index df_market_summary để dễ dàng truy cập
    if df_market_summary.empty:
        st.error("Không có dữ liệu tăng trưởng thị trường cho kỳ này.")
        df_market_indexed = pd.DataFrame()
    else:
        # Đảm bảo index là tên chỉ tiêu ngắn gọn, nhất quán với logic tính toán
        df_market_indexed = df_market_summary.set_index('Chỉ tiêu')
    
    
    # Định nghĩa các chỉ tiêu cần hiển thị và tên ngắn gọn
    metrics_to_show = {
        'Doanh thu thuần / Thu nhập lãi thuần': 'Doanh Thu Thuần',
        'Tổng lợi nhuận kế toán trước thuế': 'LN Trước Thuế',
        'Lợi nhuận sau thuế thu nhập doanh nghiệp': 'LN Sau Thuế'
    }
    
    # Chia 6 cột cho 6 KPIs (YoY và YTD của 3 metrics)
    col_yoy_dt, col_ytd_dt, col_yoy_lntt, col_ytd_lntt, col_yoy_lnst, col_ytd_lnst = st.columns(6)

    # Lặp qua các chỉ tiêu và hiển thị KPI
    for i, (metric_long, metric_short) in enumerate(metrics_to_show.items()):
        
        if metric_long in df_market_indexed.index:
            row = df_market_indexed.loc[metric_long]
            
            # Hàm phụ trợ để làm sạch giá trị string (vd: "6.29%") thành float
            def clean_value_for_kpi(value):
                try:
                    return float(value.strip('%'))
                except (ValueError, AttributeError):
                    return 0.0
            
            # --- YoY KPI ---
            yoy_value_str = row.get('YoY (All) %', 'N/A')
            yoy_value_float = clean_value_for_kpi(yoy_value_str)
            with [col_yoy_dt, col_yoy_lntt, col_yoy_lnst][i]:
                style_kpi_card(
                    f"{metric_short} (YoY)", 
                    yoy_value_str, 
                    yoy_value_float, 
                    color_pos='#03943F', 
                    color_neg='#B22F16' 
                )
                
            # --- YTD KPI ---
            ytd_value_str = row.get('YTD (All) %', 'N/A')
            ytd_value_float = clean_value_for_kpi(ytd_value_str)
            with [col_ytd_dt, col_ytd_lntt, col_ytd_lnst][i]:
                style_kpi_card(
                    f"{metric_short} (YTD)", 
                    ytd_value_str, 
                    ytd_value_float, 
                    color_pos='#03943F', 
                    color_neg='#B22F16'
                )
        else:
            # Xử lý trường hợp không có dữ liệu (Hiển thị N/A)
            with [col_yoy_dt, col_yoy_lntt, col_yoy_lnst][i]:
                style_kpi_card(f"{metric_short} (YoY)", "N/A", 0, color_pos='#636466', color_neg='#636466')
            with [col_ytd_dt, col_ytd_lntt, col_ytd_lnst][i]:
                style_kpi_card(f"{metric_short} (YTD)", "N/A", 0, color_pos='#636466', color_neg='#636466')


    st.markdown("---") # Dấu phân cách


    # 2. BẢNG THỐNG KÊ (Di chuyển xuống dưới và chia cột)
    st.markdown("<h3 style='color: #1569B4;'>📊 Tình hình Công bố Báo cáo</h3>", unsafe_allow_html=True)
    
    col_stats_data = st.container()

    # Lấy dữ liệu stats
    if df_market_stats_historical is not None:
        stats_display = df_market_stats_historical[
            (df_market_stats_historical['Nam'] == select_year) &
            (df_market_stats_historical['Quy'] == select_quarter)
        ]
    else:
        stats_display = pd.DataFrame()
        
    with col_stats_data:
        st.subheader(f"Thống kê Thị trường {select_quarter} - {select_year}")
        if stats_display.empty:
            st.warning("Không có dữ liệu thống kê cho kỳ này.")
        else:
            # Transpose và áp dụng styling
            st.dataframe(style_table_header(stats_display.T), use_container_width=True) # Sửa: .T (Transpose)

    st.markdown("---")


    # 3. BIỂU ĐỒ XU HƯỚNG
    st.subheader("Biểu đồ Xu hướng Tăng trưởng")
    
    col1, col2 = st.columns(2)
    with col1:
        metric_v5 = st.selectbox(
            "Chọn chỉ tiêu (Biểu đồ xu hướng):",
            ('LoiNhuanSauThue', 'LoiNhuanTruocThue', 'DoanhThuThuan'),
            format_func=lambda x: "LN Sau thuế" if x == 'LoiNhuanSauThue' else "LN Trước thuế" if x == 'LoiNhuanTruocThue' else 'Doanh thu thuần',
            key='v5_metric'
        )
    with col2:
        periods_v5 = st.slider(
            "Chọn số kỳ (Biểu đồ xu hướng):",
            min_value=4, max_value=20, value=12, key='v5_periods'
        )

    
    fig_v5 = utils.generate_professional_growth_chart_v5(
        df_merged,
        metric_to_plot=metric_v5,
        select_year=select_year,
        select_quarter=select_quarter,
        lookback_periods=periods_v5
    )

    if fig_v5:
        st.plotly_chart(fig_v5, use_container_width=True)
    else:
        st.warning(f"Không thể tạo Biểu đồ Xu hướng V5 cho {metric_v5}. Hãy kiểm tra log.")

# -----------------------------------------------
# TAB 2: CHI TIẾT NGÀNH
# -----------------------------------------------
with tab2:
    st.header(f"Tăng trưởng Chi tiết theo Ngành ({select_quarter} {select_year})")
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        metric_industry = st.selectbox(
            "Chọn chỉ tiêu:",
            ('Doanh thu thuần / Thu nhập lãi thuần', 'Tổng lợi nhuận kế toán trước thuế', 'Lợi nhuận sau thuế thu nhập doanh nghiệp'),
            key='industry_metric'
        )
    with col_i2:
        growth_type_industry = st.selectbox(
            "Chọn loại tăng trưởng:",
            ('YoY (All) %', 'YTD (All) %', 'QoQ (All) %', 'YoY (Same Firms) %'),
            key='industry_growth_type'
        )

    # Gọi hàm vẽ biểu đồ bar 
    fig_industry = utils.plot_growth_by_industry_plotly_v5(
        df_industry,
        growth_type=growth_type_industry,
        metric=metric_industry
    )
    
    if fig_industry:
        st.plotly_chart(fig_industry, use_container_width=True)
    
    with st.expander("Xem dữ liệu chi tiết (Bảng)"):
        st.dataframe(df_industry, use_container_width=True)

# -----------------------------------------------
# TAB 3: TOP CỔ PHIẾU
# -----------------------------------------------
with tab3:
    st.header(f"Top 20 Cổ phiếu ({select_quarter} {select_year})")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        metric_top = st.selectbox(
            "Chọn chỉ tiêu phân tích:",
            ('LoiNhuanSauThue', 'LoiNhuanTruocThue', 'DoanhThuThuan'),
            format_func=lambda x: "Lợi nhuận sau thuế" if x == 'LoiNhuanSauThue' else "Lợi nhuận trước thuế" if x == 'LoiNhuanTruocThue' else 'Doanh thu thuần',
            key='top_metric'
        )
    with col_t2:
        cap_filter = st.selectbox(
            "Lọc theo Vốn hóa:",
            ("Tất cả", "BigCap", "MidCap", "SmallCap"),
            key='cap_filter'
        )
        
    metric_options_map = {
        'DoanhThuThuan': 'Doanh thu thuần',
        'LoiNhuanTruocThue': 'Lợi nhuận trước thuế',
        'LoiNhuanSauThue': 'Lợi nhuận sau thuế'
    }

    # Gọi hàm hiển thị Top/Bottom (dùng top_results đã được lọc)
    utils.display_top_bottom_with_cap_filter(
        st=st, 
        top_results=top_results,
        metric_col=metric_top,
        current_quarter=select_quarter,
        selected_cap_group=cap_filter,
        metric_options=metric_options_map
    )
