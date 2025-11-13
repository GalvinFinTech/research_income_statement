# streamlit_app.py (Phiên bản 2.0 - Hybrid)
import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import numpy as np

# Import file utils của bạn
try:
    import utils_optimized as utils 
except ImportError:
    st.error("LỖI: Không tìm thấy file utils_optimized.py!")
    st.stop()

# -----------------------------------------------------------------
# CẤU HÌNH APP
# -----------------------------------------------------------------
DATA_DIR = "data_cache" 

# Tự động lấy Năm và Quý hiện tại (để làm giá trị mặc định cho bộ lọc)
def get_current_period():
    today = datetime.date.today()
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1
    # Logic lùi 1 quý (vì Q4 2025 có thể chưa có dữ liệu hoàn chỉnh)
    if current_quarter == 1:
        return current_year - 1, "Q4"
    else:
        # Giả định dữ liệu quý trước là đầy đủ nhất
        return current_year, f"Q{current_quarter - 1}"

DEFAULT_YEAR, DEFAULT_QUARTER = get_current_period()
print(f"Kỳ mặc định được chọn: {DEFAULT_QUARTER} {DEFAULT_YEAR}")

# -----------------------------------------------------------------
# TẢI DỮ LIỆU TỪ FILE (CỰC NHANH VỚI CACHE)
# -----------------------------------------------------------------
@st.cache_data(ttl=600) # Cache dữ liệu trong 10 phút
def load_data_from_cache():
    print(f"[{datetime.datetime.now()}] Đang tải dữ liệu từ cache (Hybrid)...")
    try:
        merged_file = os.path.join(DATA_DIR, "df_merged.parquet")
        stats_file = os.path.join(DATA_DIR, "df_market_stats_historical.parquet")
        
        # Kiểm tra file đã tồn tại chưa
        if not os.path.exists(merged_file) or not os.path.exists(stats_file):
            st.error(f"Lỗi: Không tìm thấy file cache trong '{DATA_DIR}'.")
            st.info("Hãy đảm bảo rằng tác vụ chạy nền (run_batch_job.py) phiên bản 2.0 đã chạy thành công ít nhất một lần.")
            return None, None, None

        last_updated_time = os.path.getmtime(merged_file)
        last_updated_str = datetime.datetime.fromtimestamp(last_updated_time).strftime('%Y-%m-%d %H:%M:%S')

        df_merged = pd.read_parquet(merged_file)
        df_market_stats_historical = pd.read_parquet(stats_file)
        
        print("Tải cache (Hybrid) thành công!")
        return last_updated_str, df_merged, df_market_stats_historical
    
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi đọc file cache: {e}")
        return None, None, None

# -----------------------------------------------------------------
# CÁC HÀM TÍNH TOÁN ON-THE-FLY (CỰC NHANH)
# -----------------------------------------------------------------
@st.cache_data(ttl=600)
def calculate_report_for_period(_df_merged, year, quarter):
    """
    Chạy tất cả các hàm phân tích (industry, top_10) cho kỳ được chọn.
    Streamlit sẽ cache lại kết quả này.
    """
    print(f"[{datetime.datetime.now()}] Đang tính toán on-the-fly cho {year}-{quarter}...")
    
    # 1. Tính tăng trưởng ngành
    df_industry = utils.calculate_industry_growth_rates_abs_base(
        _df_merged,
        industry_col='Phân ngành - ICB L2',
        filter_year=year,
        filter_quarter=int(quarter[1:])
    )
    
    # 2. Phân tích Top 20
    top_results = utils.analyze_top_10_stocks(
        _df_merged,
        current_year=year,
        current_quarter=quarter,
        top_n=20
    )
    
    # 3. Lấy tóm tắt thị trường (chỉ phần tăng trưởng)
    df_market_summary = df_industry[df_industry['Phân loại'] == 'Toàn thị trường']
    
    print(f"   ...Tính toán on-the-fly cho {year}-{quarter} HOÀN TẤT.")
    return df_industry, top_results, df_market_summary

# -----------------------------------------------------------------
# BẮT ĐẦU VẼ GIAO DIỆN APP
# -----------------------------------------------------------------
st.set_page_config(layout="wide")

# Tải dữ liệu chính
data = load_data_from_cache()
last_updated, df_merged, df_market_stats_historical = data

# Tiêu đề và thời gian cập nhật
st.title("📈 Báo cáo Tăng trưởng Thị trường (Hybrid)")
if last_updated:
    st.caption(f"Dữ liệu gốc được cập nhật lần cuối lúc: **{last_updated}** (Giờ máy chủ)")

# Nếu tải dữ liệu thất bại, dừng app ở đây
if df_merged is None:
    st.stop()

# -----------------------------------------------
# BỘ LỌC CHÍNH (Sidebar)
# -----------------------------------------------
st.sidebar.header("Bộ lọc Báo cáo")

# Lấy danh sách Năm và Quý từ dữ liệu đã có
available_years = sorted(df_merged['Nam'].unique(), reverse=True)
available_quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# Tìm index của giá trị mặc định
try:
    year_index = available_years.index(DEFAULT_YEAR)
except ValueError:
    year_index = 0 # Nếu không tìm thấy, lấy năm mới nhất

try:
    quarter_index = available_quarters.index(DEFAULT_QUARTER)
except ValueError:
    quarter_index = 0

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

st.sidebar.info(f"Đang hiển thị báo cáo cho: **{select_quarter} {select_year}**")

# -----------------------------------------------
# TÍNH TOÁN ON-THE-FLY DỰA TRÊN BỘ LỌC
# -----------------------------------------------
# Streamlit sẽ chỉ chạy lại khi (select_year, select_quarter) thay đổi
# và nó sẽ dùng lại kết quả cache nếu người dùng chọn lại.
df_industry, top_results, df_market_summary = calculate_report_for_period(
    df_merged, select_year, select_quarter
)

# -----------------------------------------------
# TAB 1: TỔNG QUAN THỊ TRƯỜNG
# -----------------------------------------------
tab1, tab2, tab3 = st.tabs(["Tổng quan Thị trường", "Chi tiết Ngành", "Top Cổ phiếu"])

with tab1:
    st.header(f"Tổng quan Tăng trưởng {select_quarter} {select_year}")
    
    st.subheader("Thống kê Thị trường (Tỷ lệ công bố)")
    if df_market_stats_historical is not None:
        stats_display = df_market_stats_historical[
            (df_market_stats_historical['Nam'] == select_year) &
            (df_market_stats_historical['Quy'] == select_quarter)
        ]
        if stats_display.empty:
            st.warning("Không có dữ liệu thống kê cho kỳ này.")
        else:
            st.dataframe(stats_display)
    
    st.subheader("Tăng trưởng Toàn thị trường (cho kỳ đã chọn)")
    st.dataframe(df_market_summary)

    # Biểu đồ V5 (biểu đồ xu hướng) không bị ảnh hưởng bởi bộ lọc Năm/Quý
    st.header("Biểu đồ Xu hướng Tăng trưởng")
    
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

    # --- TRONG streamlit_app.py, Ở TAB 1 ---

    # Đổi tên hàm gọi (nếu bạn đã đổi thành v6)
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

    # Gọi hàm vẽ biểu đồ bar (dùng df_industry đã được lọc)
    fig_industry = utils.plot_growth_by_industry_plotly_v5(
        df_industry,
        growth_type=growth_type_industry,
        metric=metric_industry
    )
    
    if fig_industry:
        st.plotly_chart(fig_industry, use_container_width=True)
    
    with st.expander("Xem dữ liệu chi tiết (Bảng)"):
        st.dataframe(df_industry)

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