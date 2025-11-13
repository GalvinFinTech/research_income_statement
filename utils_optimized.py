# utils.py
import pandas as pd
import numpy as np
import pyodbc
from typing import List, Dict, Optional, Union
import plotly.express as px
import plotly.graph_objects as go


# Function to get latest reporters with stats
def get_latest_reporters_with_stats(sql_connection_string: str, current_year: int, current_quarter: str, term_type_filter: int = -1) -> pd.DataFrame:
    sql_query = f"""
    DECLARE @CurrentYear int = {current_year};
    DECLARE @CurrentQuarter varchar(2) = '{current_quarter}';
    DECLARE @TermTypeFilter int = {term_type_filter};

    SELECT 
        c.CompanyCode AS MaCoPhieu,
        c.FullName AS TenCongTy,
        bt.Description AS LoaiHinhCongTy,
        ci.Name AS TenNganh,
        ci2.Name AS TenNganhCon,
        rd.YearPeriod AS NamBaoCao,
        rt.TermCode AS KyBaoCao,
        rd.LastUpdate AS NgayCongBo,
        rd.MarketCap AS VonHoa 
    FROM VSTDataFeed.dbo.Company c WITH (NOLOCK)
    INNER JOIN VSTDataFeed.dbo.ReportData rd WITH (NOLOCK) 
        ON rd.CompanyID = c.CompanyID
    INNER JOIN VSTDataFeed.dbo.ReportTerm rt WITH (NOLOCK) 
        ON rt.ReportTermID = rd.ReportTermID
    LEFT JOIN VSTDataFeed.dbo.BusinessType bt WITH (NOLOCK) 
        ON bt.BusinessTypeID = c.CompanyType + 1
    LEFT JOIN VSTDataFeed.dbo.ChannelIndustry ci WITH (NOLOCK) 
        ON ci.IndustryID = c.IndustryID
    LEFT JOIN VSTDataFeed.dbo.ChannelIndustry ci2 WITH (NOLOCK) 
        ON ci2.IndustryID = c.SubIndustry
    WHERE 
        c.Status = 1 
        AND c.CatID IN (1, 2, 5)
        AND rd.IsUnited IN (0, 1)
        AND rd.YearPeriod = @CurrentYear
        AND (
            (@TermTypeFilter = 2 AND rt.TermCode = @CurrentQuarter) 
            OR (@TermTypeFilter = 1 AND rt.ReportTermTypeID = 1) 
            OR @TermTypeFilter = -1
        )
    GROUP BY 
        c.CompanyCode, c.FullName, bt.Description, ci.Name, ci2.Name,
        rd.YearPeriod, rt.TermCode, rd.LastUpdate, rd.MarketCap
    ORDER BY rd.LastUpdate DESC;
    """
    try:
        conn = pyodbc.connect(sql_connection_string)
        df = pd.read_sql(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return pd.DataFrame()

# Function to get core financials by list
# Function to get core financials by list
def get_core_financials_by_list(sql_connection_string: str, stock_codes_list: List[str], start_year: int = 2020, unit: int = 1000000) -> pd.DataFrame:
    if not stock_codes_list:
        print("⚠️ Empty stock codes list. Skipping query.")
        return pd.DataFrame()

    codes_string_sql = ', '.join([f"'{code}'" for code in stock_codes_list])
    
    sql_template = f"""
    DECLARE @Unit BIGINT = {unit};
    DECLARE @StartYear INT = {start_year};
    DECLARE @VonHoaUnit BIGINT = {unit};

    WITH CompanyInfo AS (
        SELECT CompanyID, CompanyCode, CompanyType, IndustryID, FullName --- # THÊM VÀO ---
        FROM VSTDataFeed.dbo.Company ci WITH (NOLOCK)
        WHERE ci.Status = 1 AND ci.CatID IN (1, 2, 5)
          AND ci.CompanyCode IN ({codes_string_sql})
    ),
    RankedReports AS (
        SELECT
            rd.ReportDataID, rd.CompanyID, rd.YearPeriod, rt.TermCode, rd.IsUnited,
            ROW_NUMBER() OVER (
                PARTITION BY rd.CompanyID, rd.YearPeriod, rt.ReportTermID
                ORDER BY rd.IsUnited ASC, rd.ReportDataID DESC
            ) AS Rank
        FROM VSTDataFeed.dbo.ReportData rd WITH (NOLOCK)
        INNER JOIN CompanyInfo ci ON ci.CompanyID = rd.CompanyID
        INNER JOIN VSTDataFeed.dbo.ReportTerm rt WITH (NOLOCK) ON rt.ReportTermID = rd.ReportTermID
        WHERE rt.ReportTermTypeID = 2 AND rd.YearPeriod >= @StartYear AND rd.IsUnited IN (0, 1)
    ),
    MarketCapLatestRanked AS (
        SELECT
            fid.CompanyID,
            ROUND(ISNULL(fid.MarketCap, 0) / @VonHoaUnit, 2) AS VonHoaMoiNhat,
            ROW_NUMBER() OVER (
                PARTITION BY fid.CompanyID
                ORDER BY fid.TradingDate DESC
            ) AS Rank
        FROM VSTDataFeed.dbo.FinanceIndexDaily fid WITH (NOLOCK)
        INNER JOIN CompanyInfo ci ON ci.CompanyID = fid.CompanyID
        WHERE ISNULL(fid.MarketCap, 0) > 0
    ),
    MarketCapLatest AS (
        SELECT CompanyID, VonHoaMoiNhat
        FROM MarketCapLatestRanked
        WHERE Rank = 1
    ),
    FinancialData AS (
        SELECT
            ci.CompanyCode AS MaCoPhieu,
            ci.FullName AS TenCongTy, --- # THÊM VÀO ---
            rr.YearPeriod AS Nam,
            rr.TermCode AS Quy,
            CASE rr.IsUnited WHEN 0 THEN N'HN' ELSE N'ĐL' END AS LoaiBaoCao,
            COALESCE(ns.NormName, rn.Name, n.NormName) AS TenChiTieu,
            ROUND(ISNULL(rdd.Value, 0) / @Unit, 2) AS GiaTri,
            ISNULL(mcl.VonHoaMoiNhat, 0.00) AS VonHoa
        FROM RankedReports rr
        INNER JOIN CompanyInfo ci ON ci.CompanyID = rr.CompanyID
        INNER JOIN VSTDataFeed.dbo.ReportDataDetails rdd WITH (NOLOCK)
            ON rdd.ReportDataID = rr.ReportDataID
        INNER JOIN VSTDataFeed.dbo.ReportNorm rn WITH (NOLOCK)
            ON rn.ReportNormID = rdd.ReportNormID
        INNER JOIN VSTDataFeed.dbo.ReportComponents rcp WITH (NOLOCK)
            ON rn.ReportComponentID = rcp.ReportComponentID
        INNER JOIN VSTDataFeed.dbo.ReportComponentTypes rct WITH (NOLOCK)
            ON rcp.ReportComponentTypeID = rct.ReportComponentTypeID
        LEFT JOIN VSTDataFeed.dbo.Norms n WITH (NOLOCK) ON n.NormID = rn.NormID
        LEFT JOIN VSTDataFeed.dbo.NormSpecial ns WITH (NOLOCK)
            ON ( (ci.CompanyType = 0 AND rn.ReportNormID = ns.ReportNormID_CP) OR
                 (ci.CompanyType = 1 AND rn.ReportNormID = ns.ReportNormID_CK) OR
                 (ci.CompanyType = 2 AND rn.ReportNormID = ns.ReportNormID_NH) OR
                 (ci.CompanyType = 3 AND rn.ReportNormID = ns.ReportNormID_Q)  OR
                 (ci.CompanyType = 4 AND rn.ReportNormID = ns.ReportNormID_BH) )
        LEFT JOIN MarketCapLatest mcl ON mcl.CompanyID = ci.CompanyID
        WHERE rr.Rank = 1 AND rct.Code = 'KQ'
            AND COALESCE(ns.NormName, rn.Name, n.NormName) IN (
                N'Doanh thu thuần', N'3. Doanh thu thuần', N'3. Doanh thu thuần về bán hàng và cung cấp dịch vụ', N'3, Doanh thu thuần về hoạt động kinh doanh(10=01-02)',
                N'III. Thu nhập lãi thuần (I-II)',
                N'13, Tổng lợi nhuận kế toán trước thuế (50=30+40+41)', N'15. Tổng lợi nhuận kế toán trước thuế', N'26. Tổng lợi nhuận kế toán trước thuế', N'Tổng lợi nhuận trước thuế thu nhập doanh nghiệp', N'IX. TỔNG LỢI NHUẬN KẾ TOÁN TRƯỚC THUẾ (70+80)', N'XI. Tổng lợi nhuận trước thuế (IX-X)', N'III. Lợi nhuận trước thuế',
                N'16, Lợi nhuận sau thuế thu nhập doanh nghiệp (60=50-51-52)', N'17. Lợi nhuận sau thuế thu nhập doanh nghiệp', N'29. Lợi nhuận sau thuế thu nhập doanh nghiệp', N'IV. Lợi nhuận sau thuế', N'XI. LỢI NHUẬN KẾ TOÁN SAU THUẾ TNDN (90-100)', N'XIII. Lợi nhuận sau thuế (XI-XII)', N'18.2 Lợi nhuận sau thuế của cổ đông của Công ty mẹ', N'31. Lợi nhuận sau thuế của cổ đông của Công ty mẹ', N'VII. Lợi nhuận sau thuế của cổ đông công ty mẹ', N'XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV)'
            )
    )
    SELECT
        MaCoPhieu, Nam, Quy, LoaiBaoCao,
        MAX(TenCongTy) AS TenCongTy, --- # THÊM VÀO ---
        MAX(VonHoa) AS VonHoa,
        ISNULL(MAX(CASE WHEN TenChiTieu IN (N'Doanh thu thuần', N'3. Doanh thu thuần', N'3. Doanh thu thuần về bán hàng và cung cấp dịch vụ', N'3, Doanh thu thuần về hoạt động kinh doanh(10=01-02)', N'III. Thu nhập lãi thuần (I-II)') THEN GiaTri END), 0) AS DoanhThuThuan,
        ISNULL(MAX(CASE WHEN TenChiTieu IN (N'13, Tổng lợi nhuận kế toán trước thuế (50=30+40+41)', N'15. Tổng lợi nhuận kế toán trước thuế', N'26. Tổng lợi nhuận kế toán trước thuế', N'Tổng lợi nhuận trước thuế thu nhập doanh nghiệp', N'IX. TỔNG LỢI NHUẬN KẾ TOÁN TRƯỚC THUẾ (70+80)', N'XI. Tổng lợi nhuận trước thuế (IX-X)', N'III. Lợi nhuận trước thuế') THEN GiaTri END), 0) AS LoiNhuanTruocThue,
        ISNULL(MAX(CASE WHEN TenChiTieu IN (N'16, Lợi nhuận sau thuế thu nhập doanh nghiệp (60=50-51-52)', N'17. Lợi nhuận sau thuế thu nhập doanh nghiệp', N'29. Lợi nhuận sau thuế thu nhập doanh nghiệp', N'IV. Lợi nhuận sau thuế', N'XI. LỢI NHUẬN KẾ TOÁN SAU THUẾ TNDN (90-100)', N'XIII. Lợi nhuận sau thuế (XI-XII)', N'18.2 Lợi nhuận sau thuế của cổ đông của Công ty mẹ', N'31. Lợi nhuận sau thuế của cổ đông của Công ty mẹ', N'VII. Lợi nhuận sau thuế của cổ đông công ty mẹ', N'XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV)') THEN GiaTri END), 0) AS LoiNhuanSauThue
    FROM FinancialData
    GROUP BY MaCoPhieu, Nam, Quy, LoaiBaoCao
    ORDER BY MaCoPhieu ASC, Nam DESC,
        CASE Quy WHEN 'Q4' THEN 1 WHEN 'Q3' THEN 2 WHEN 'Q2' THEN 3 WHEN 'Q1' THEN 4 ELSE 5 END ASC;
    """
    try:
        conn = pyodbc.connect(sql_connection_string, fast_executemany=True)
        print(f"Executing V3 query (prioritizing consolidated reports + market cap) for {len(stock_codes_list)} stocks...")
        df = pd.read_sql(sql_template, conn)
        conn.close()
        print(f"✅ Query successful! Retrieved {len(df):,} rows.")
        return df
    except Exception as e:
        print(f"❌ Error executing SQL V3 query: {e}")
        try: conn.close()
        except: pass
        return pd.DataFrame()


# Add this new function just before calculate_industry_growth_rates_abs_base
def calculate_growth_rate_abs_base(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate absolute growth rate for a pandas Series with (current - previous) / |previous|.
    
    Parameters:
    - series: pandas.Series with multi-index (e.g., ['Nam', 'Quy_int']) or single index.
    - periods: Number of periods to shift for calculating growth (e.g., 1 for QoQ, 4 for YoY).
    
    Returns:
    - pandas.Series with growth rates (as decimals).
    """
    previous = series.shift(periods=periods)
    numerator = series - previous
    denominator = previous.abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        growth = np.divide(
            numerator,
            denominator,
            out=np.full_like(series, np.nan, dtype=np.float64),
            where=(denominator != 0) & (~np.isnan(denominator))
        )
    return growth


def calculate_industry_growth_rates_abs_base(df, industry_col='Phân ngành - ICB L2', filter_year=2025, filter_quarter=3):
    print(f"\nCalculating growth report for Q{filter_quarter} {filter_year}")
    required_cols = [industry_col, 'Quy', 'Nam', 'MaCoPhieu']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns {missing_cols}. Available: {list(df.columns)}")
        return None

    df_copy = df.copy()
    try:
        df_copy['Quy_int'] = df_copy['Quy'].astype(str).str.replace('Q', '').astype(int)
    except (ValueError, KeyError) as e:
        print(f"ERROR: Unable to process 'Quy' column: {e}")
        return None

    q_filter = int(str(filter_quarter)[-1])
    df_copy = df_copy[
        (df_copy['Nam'] < filter_year) |
        ((df_copy['Nam'] == filter_year) & (df_copy['Quy_int'] <= q_filter))
    ]
    if df_copy.empty:
        print("ERROR: No data after filtering.")
        return None

    mapping = {
        'DoanhThuThuan': 'Doanh thu thuần / Thu nhập lãi thuần',
        'LoiNhuanTruocThue': 'Tổng lợi nhuận kế toán trước thuế',
        'LoiNhuanSauThue': 'Lợi nhuận sau thuế thu nhập doanh nghiệp'
    }

    def format_percent(val):
        return 'N/A' if pd.isna(val) or np.isinf(val) else f"{val * 100:.2f}%"

    # --- HÀM TÍNH TOÁN (All) MỚI ---
    # Logic "thủ công", không dùng pivot/shift
    def calculate_simple_growth(current_val, previous_val):
        if previous_val is None or current_val is None or pd.isna(previous_val) or pd.isna(current_val):
            return np.nan
        
        denominator = abs(previous_val)
        if denominator == 0:
            return np.nan # Không thể so sánh với 0
            
        return (current_val - previous_val) / denominator

    # --- HÀM TÓM TẮT ĐÃ SỬA LỖI ---
    def _internal_summarize(df_subset, label, current_year, current_quarter, mapping):
        q_int = int(str(current_quarter)[-1])
        PQ_Y, PQ_Q = (current_year - 1, 4) if q_int == 1 else (current_year, q_int - 1)
        PY_Y, PY_Q = current_year - 1, q_int
        
        # Danh sách quý YTD hiện tại (ví dụ: [1, 2, 3] cho Q3)
        YTD_Q_list = list(range(1, q_int + 1))
        
        rows = []
        for metric_col, display_name in mapping.items():
            if metric_col not in df_subset.columns:
                print(f"⚠️ Column '{metric_col}' not found for '{label}'.")
                continue
            
            # Chỉ làm việc với dữ liệu có giá trị
            df_metric = df_subset.dropna(subset=[metric_col])
            if df_metric.empty:
                continue
            
            # --- LOGIC TÍNH (All) MỚI ---
            # 1. Pivot để lấy tổng (SUM) cho mỗi kỳ
            df_agg_table = df_metric.pivot_table(
                values=metric_col, index=['Nam', 'Quy_int'], aggfunc='sum'
            )
            
            # Helper để lấy giá trị tổng, trả về None nếu thiếu
            def get_sum(year, quarter):
                key = (year, quarter)
                if key in df_agg_table.index:
                    return df_agg_table.loc[key, metric_col]
                return None # Trả về None (không phải 0) nếu thiếu

            # 2. Lấy các giá trị tổng
            curr_q_val = get_sum(current_year, q_int)
            prev_q_qoq_val = get_sum(PQ_Y, PQ_Q)
            prev_q_yoy_val = get_sum(PY_Y, PY_Q)
            
            # 3. Tính tổng YTD (bằng cách lọc và sum, không dùng cumsum)
            curr_ytd_val = df_metric[
                (df_metric['Nam'] == current_year) & (df_metric['Quy_int'].isin(YTD_Q_list))
            ][metric_col].sum()
            
            prev_ytd_val = df_metric[
                (df_metric['Nam'] == PY_Y) & (df_metric['Quy_int'].isin(YTD_Q_list))
            ][metric_col].sum()
            
            # 4. Tính toán tăng trưởng (All)
            qoq_val = calculate_simple_growth(curr_q_val, prev_q_qoq_val)
            yoy_val = calculate_simple_growth(curr_q_val, prev_q_yoy_val)
            ytd_val = calculate_simple_growth(curr_ytd_val, prev_ytd_val)
            
            # --- LOGIC TÍNH (Same Firms) --- (Vẫn giữ nguyên vì logic này đã đúng)
            df_curr = df_metric[(df_metric['Nam'] == current_year) & (df_metric['Quy_int'] == q_int)]
            df_prev_yoy = df_metric[(df_metric['Nam'] == PY_Y) & (df_metric['Quy_int'] == PY_Q)]
            
            common_yoy = set(df_curr['MaCoPhieu']).intersection(df_prev_yoy['MaCoPhieu'])
            yoy_same_firms_val = np.nan
            if common_yoy:
                curr_sum_sf = df_curr[df_curr['MaCoPhieu'].isin(common_yoy)][metric_col].sum()
                prev_sum_sf = df_prev_yoy[df_prev_yoy['MaCoPhieu'].isin(common_yoy)][metric_col].sum()
                yoy_same_firms_val = calculate_simple_growth(curr_sum_sf, prev_sum_sf)

            # --- Đếm số công ty ---
            n_curr = df_curr['MaCoPhieu'].nunique()
            n_prev_yoy = df_prev_yoy['MaCoPhieu'].nunique()
            n_prev_qoq = df_metric[
                (df_metric['Nam'] == PQ_Y) & (df_metric['Quy_int'] == PQ_Q)
            ]['MaCoPhieu'].nunique()
            
            rows.append({
                'Phân loại': label,
                'Chỉ tiêu': display_name,
                'QoQ (All) %': format_percent(qoq_val),
                'YoY (All) %': format_percent(yoy_val),
                'YoY (Same Firms) %': format_percent(yoy_same_firms_val),
                'YTD (All) %': format_percent(ytd_val),
                'Số công ty có dữ liệu Q_prev(QoQ)': n_prev_qoq,
                'Số công ty có dữ liệu Q_prev(YoY)': n_prev_yoy,
                'Số công ty có dữ liệu Q_curr': n_curr
            })
        return pd.DataFrame(rows)

    print("Calculating: Market-wide...")
    df_market = _internal_summarize(df_copy, 'Toàn thị trường', filter_year, q_filter, mapping)
    print(f"Calculating: {industry_col}...")
    df_copy[industry_col] = df_copy[industry_col].fillna('Chưa phân loại')
    industry_results = [
        _internal_summarize(df_copy[df_copy[industry_col] == ind], ind, filter_year, q_filter, mapping)
        for ind in sorted(df_copy[industry_col].unique())
    ]
    df_industry = pd.concat(industry_results, ignore_index=True) if industry_results else pd.DataFrame()
    df_final = pd.concat([df_market, df_industry], ignore_index=True)
    print("Report completed.")
    return df_final


def plot_growth_by_industry_plotly_v5(df_industry_summary, growth_type='YoY (All) %', metric='Doanh thu thuần / Thu nhập lãi thuần'):
    valid_types = ['QoQ (All) %', 'YoY (All) %', 'YoY (Same Firms) %', 'YTD (All) %']
    if growth_type not in valid_types:
        raise ValueError(f"❌ Invalid growth type: {valid_types}")

    df_plot = df_industry_summary.copy()
    df_plot[growth_type] = (
        df_plot[growth_type]
        .astype(str)
        .str.replace('%', '')
        .replace(['N/A (Base 0)', 'N/A', ''], np.nan)
        .astype(float)
    )
    df_metric = df_plot[df_plot['Chỉ tiêu'].str.strip() == metric.strip()].dropna(subset=[growth_type])

    if df_metric.empty:
        print(f"⚠️ No data for '{metric}' ({growth_type}). Available: {df_plot['Chỉ tiêu'].str.strip().unique()}")
        return None

    col_firms = {
        'QoQ': 'Số công ty có dữ liệu Q_prev(QoQ)',
        'YoY': 'Số công ty có dữ liệu Q_prev(YoY)',
        'YTD': 'Số công ty có dữ liệu Q_curr'
    }.get(growth_type.split()[0], 'Số công ty có dữ liệu Q_curr')

    df_metric = df_metric.sort_values(growth_type, ascending=True)
    market_row = df_metric[df_metric['Phân loại'] == 'Toàn thị trường']
    industry_rows = df_metric[df_metric['Phân loại'] != 'Toàn thị trường']
    df_metric = pd.concat([industry_rows, market_row]).reset_index(drop=True)

    # Gán màu cho từng loại
    df_metric['color_category'] = (
        df_metric[growth_type]
        .mask(df_metric['Phân loại'] == 'Toàn thị trường', 'Toàn thị trường')
        .mask(df_metric[growth_type] >= 0, 'Tăng')
        .mask(df_metric[growth_type] < 0, 'Giảm')
    )

    color_map = {'Tăng': '#1f77b4', 'Giảm': '#ff7f0e', 'Toàn thị trường': '#2ca02c'}

    fig = px.bar(
        df_metric,
        x=growth_type, y='Phân loại', orientation='h',
        color='color_category', color_discrete_map=color_map,
        text=df_metric[growth_type].map(lambda x: f"{x:.1f}%"),
        hover_data=['Phân loại', 'Chỉ tiêu', growth_type, col_firms],
        title=f"Tăng trưởng {metric} theo ngành - {growth_type}",
        template='plotly_white'
    )

    # Cập nhật định dạng biểu đồ
    fig.update_traces(
        textposition='outside',
        insidetextanchor='middle',
        textfont=dict(size=12, color='black'),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% tăng trưởng<extra></extra>",
        marker_line_color=[
            'black' if x == 'Toàn thị trường' else 'rgba(0,0,0,0)'
            for x in df_metric['Phân loại']
        ],
        marker_line_width=[
            1.5 if x == 'Toàn thị trường' else 0
            for x in df_metric['Phân loại']
        ]
    )

    # Cập nhật layout
    fig.update_layout(
        font=dict(family="Arial", size=13, color="#2c3e50"),
        
        xaxis_title=growth_type,
        yaxis_title='Ngành',
        bargap=0.15,
        height=max(500, len(df_metric) * 40),
        margin=dict(l=120, r=60, t=100, b=60),
        legend_title_text='Xu hướng',
        legend=dict(orientation="h", y=1.05, x=0.5),
        yaxis={'categoryorder': 'array', 'categoryarray': df_metric['Phân loại'].tolist()}
    )

    # Thêm đường mốc 0%
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="gray")

    # Giới hạn trục x để tránh bị cắt mất phần nhãn
    try:
        x_min = df_metric[growth_type].min()
        x_max = df_metric[growth_type].max()
        pad = (x_max - x_min) * 0.15 if not np.isnan(x_min) and not np.isnan(x_max) else 1
        fig.update_xaxes(range=[x_min - pad, x_max + pad])
    except Exception as e:
        print(f"⚠️ Error setting x-axis range: {e}")

    print("✅ Industry growth chart generated successfully.")
    return fig


def analyze_top_10_stocks(df_merged: pd.DataFrame, current_year: int, current_quarter: str, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    print("\n--- Starting Top 10 Analysis (Optimized) ---")
    df = df_merged.copy()
    metrics = ['DoanhThuThuan', 'LoiNhuanTruocThue', 'LoiNhuanSauThue']
    q_int = int(current_quarter[1:])
    df['Quy_int'] = df['Quy'].str.replace('Q', '').astype(int)

    # Sắp xếp theo MaCoPhieu, Nam, Quy_int
    df_sorted = df.sort_values(by=['MaCoPhieu', 'Nam', 'Quy_int']).reset_index(drop=True)
    
    # Tạo đối tượng GroupBy để tái sử dụng
    df_grouped = df_sorted.groupby('MaCoPhieu')

    results = {}
    
    for metric in metrics:
        print(f"   Calculating growth for: {metric}...")
        
        # --- Tính giá trị YTD lũy kế (FIX: Tính lũy kế theo từng Năm) ---
        ytd_col = f'{metric}_YTD_Value'
        df_sorted[ytd_col] = df_sorted.groupby(['MaCoPhieu', 'Nam'])[metric].cumsum()

        # --- Tính tăng trưởng QoQ (vector hóa) ---
        qoq_col = f'{metric}_QoQ'
        prev_q = df_grouped[metric].shift(1)
        # Sử dụng (current - previous) / abs(previous)
        df_sorted[qoq_col] = calculate_growth_rate_abs_base(df_sorted[metric], periods=1)
        # Cần reset lại sau khi groupby vì shift() không giữ group
        df_sorted[qoq_col] = df_sorted[qoq_col].where(df_sorted['MaCoPhieu'] == df_sorted['MaCoPhieu'].shift(1))


        # --- Tính tăng trưởng YoY (vector hóa) ---
        yoy_col = f'{metric}_YoY'
        # Sử dụng (current - previous) / abs(previous)
        df_sorted[yoy_col] = calculate_growth_rate_abs_base(df_sorted[metric], periods=4)
        df_sorted[yoy_col] = df_sorted[yoy_col].where(df_sorted['MaCoPhieu'] == df_sorted['MaCoPhieu'].shift(4))

        # --- Tính tăng trưởng YTD (vector hóa) ---
        ytd_growth_col = f'{metric}_YTD_Growth'
        prev_ytd_val = df_grouped[ytd_col].shift(4)
        df_sorted[ytd_growth_col] = (df_sorted[ytd_col] - prev_ytd_val) / np.abs(prev_ytd_val)
        df_sorted[ytd_growth_col] = df_sorted[ytd_growth_col].where(df_sorted['MaCoPhieu'] == df_sorted['MaCoPhieu'].shift(4))
        # Thay thế inf bằng nan
        df_sorted.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Lọc dữ liệu cho quý hiện tại ---
    df_current = df_sorted[(df_sorted['Nam'] == current_year) & (df_sorted['Quy_int'] == q_int)].copy()
    
    if df_current.empty:
        print("   ⚠️ No data for current period. Returning empty results.")
        return {}

    # --- Tạo kết quả Top/Bottom ---
    growth_types = {'QoQ': '_QoQ', 'YoY': '_YoY', 'YTD': '_YTD_Growth'}
    
    for metric in metrics:
        # Giá trị Quý (Q)
        df_top = df_current.sort_values(by=metric, ascending=False).head(top_n)
        df_bottom = df_current.sort_values(by=metric, ascending=True).head(top_n)
        results[f'Top_{metric}_Q'] = df_top[['MaCoPhieu', 'TenCongTy', 'VonHoa', metric]]
        results[f'Bottom_{metric}_Q'] = df_bottom[['MaCoPhieu', 'TenCongTy', 'VonHoa', metric]]

        # Giá trị YTD
        ytd_col = f'{metric}_YTD_Value'
        df_top = df_current.sort_values(by=ytd_col, ascending=False).head(top_n)
        df_bottom = df_current.sort_values(by=ytd_col, ascending=True).head(top_n)
        results[f'Top_{metric}_YTD'] = df_top[['MaCoPhieu', 'TenCongTy', 'VonHoa', ytd_col]]
        results[f'Bottom_{metric}_YTD'] = df_bottom[['MaCoPhieu', 'TenCongTy', 'VonHoa', ytd_col]]

        # Tăng trưởng QoQ, YoY, YTD
        for label, col_suffix in growth_types.items():
            col = f'{metric}{col_suffix}'
            df_valid = df_current.dropna(subset=[col])
            df_top = df_valid.sort_values(by=col, ascending=False).head(top_n)
            df_bottom = df_valid.sort_values(by=col, ascending=True).head(top_n)
            results[f'Top_{metric}_{label}_Growth'] = df_top[['MaCoPhieu', 'TenCongTy', 'VonHoa', col]]
            results[f'Bottom_{metric}_{label}_Growth'] = df_bottom[['MaCoPhieu', 'TenCongTy', 'VonHoa', col]]
            
    print("   ✅ Top 10 Analysis completed.")
    return results



def classify_market_cap(vonhoa):
    if pd.isna(vonhoa):
        return "Unknown"
    if vonhoa >= 10_000:     # >= 10,000 tỷ
        return "BigCap"
    elif vonhoa >= 1_000:    # từ 1,000 – 10,000 tỷ
        return "MidCap"
    else:                    # < 1,000 tỷ
        return "SmallCap"

def display_top_bottom_with_cap_filter(
    st,
    top_results: dict,
    metric_col: str,
    current_quarter: str,
    selected_cap_group: str = "Tất cả",
    metric_options: dict = None
):
    """
    Hiển thị Top/Bottom theo giá trị Q, YTD, YoY và YTD Growth (%) với dữ liệu đã tính sẵn trong top_results.
    """

    vn_name = metric_options.get(metric_col, metric_col) if metric_options else metric_col

    tables = [
        ("Q", f"Giá trị Quý {current_quarter[-1]}", f"Top_{metric_col}_Q", f"Bottom_{metric_col}_Q"),
        ("YTD", "Giá trị YTD", f"Top_{metric_col}_YTD", f"Bottom_{metric_col}_YTD"),
        ("YoY", "Tăng trưởng YoY", f"Top_{metric_col}_YoY_Growth", f"Bottom_{metric_col}_YoY_Growth"),
        ("YTD_Growth", "Tăng trưởng YTD (%)", f"Top_{metric_col}_YTD_Growth", f"Bottom_{metric_col}_YTD_Growth")
    ]

    def prepare_df(df, label, val_col_name):
        if df is None or df.empty:
            return pd.DataFrame(columns=['MaCoPhieu', 'VonHoa', val_col_name])
        
        df_disp = df.copy()
        
        # Đổi tên cột giá trị thành 'Giá trị' để chuẩn hóa
        df_disp = df_disp.rename(columns={val_col_name: 'Giá trị'})
        
        # Format giá trị
        if label in ["YoY", "YTD_Growth"]:
            df_disp['Giá trị'] = df_disp['Giá trị'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        else:
            df_disp['Giá trị'] = df_disp['Giá trị'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

        # Format Vốn hóa
        if 'VonHoa' in df_disp.columns:
            df_disp['VonHoa'] = df_disp['VonHoa'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        # Sắp xếp lại cột
        return df_disp[['MaCoPhieu', 'VonHoa', 'Giá trị']]

    def style_growth(val):
        if isinstance(val, str) and val.endswith('%'):
            try:
                val_num = float(val.strip('%'))
                return 'color: green' if val_num > 0 else 'color: red' if val_num < 0 else ''
            except:
                return ''
        return ''

    for label, title, top_key, bottom_key in tables:
        
        df_top_raw = top_results.get(top_key)
        df_bottom_raw = top_results.get(bottom_key)

        # --- FIX: Áp dụng bộ lọc vốn hóa ---
        if selected_cap_group != "Tất cả":
            if df_top_raw is not None and not df_top_raw.empty:
                df_top_raw['CapGroup'] = df_top_raw['VonHoa'].apply(classify_market_cap)
                df_top_raw = df_top_raw[df_top_raw['CapGroup'] == selected_cap_group].drop(columns=['CapGroup'])
            if df_bottom_raw is not None and not df_bottom_raw.empty:
                df_bottom_raw['CapGroup'] = df_bottom_raw['VonHoa'].apply(classify_market_cap)
                df_bottom_raw = df_bottom_raw[df_bottom_raw['CapGroup'] == selected_cap_group].drop(columns=['CapGroup'])

        # Lấy tên cột giá trị (cột cuối cùng)
        val_col_name_top = df_top_raw.columns[-1] if df_top_raw is not None else 'Giá trị'
        val_col_name_bottom = df_bottom_raw.columns[-1] if df_bottom_raw is not None else 'Giá trị'
        
        df_top_disp = prepare_df(df_top_raw, label, val_col_name_top)
        df_bottom_disp = prepare_df(df_bottom_raw, label, val_col_name_bottom)

        if (df_top_disp.empty) and (df_bottom_disp.empty):
            st.markdown(f"**{title}** — Không có dữ liệu cho nhóm '{selected_cap_group}'")
            continue

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Top {title} ({vn_name})**")
            st.dataframe(
                df_top_disp.style.applymap(style_growth, subset=['Giá trị'])
                .set_table_styles([{'selector': 'th', 'props': [('background-color', '#cce5ff')]}]),
                use_container_width=True
            )

        with col2:
            st.markdown(f"**Bottom {title} ({vn_name})**")
            st.dataframe(
                df_bottom_disp.style.applymap(style_growth, subset=['Giá trị'])
                .set_table_styles([{'selector': 'th', 'props': [('background-color', '#cce5ff')]}]),
                use_container_width=True
            )



# Function to get market totals
def get_market_totals_latest(execute_sql_func, unit=1_000_000): # đổi tên để tránh trùng lặp
    query_total_companies = """
        SELECT COUNT(*) AS TotalListed
        FROM VSTDataFeed.dbo.Company
        WHERE Status = 1 AND CatID IN (1,2,5)
    """
    total_listed = execute_sql_func(query_total_companies)['TotalListed'].iloc[0]
    
    # --- OPTIMIZATION: Thêm bộ lọc c.Status, c.CatID ---
    # Lọc bớt danh sách công ty ngay từ đầu (giống query_total_companies)
    # sẽ giúp query vốn hóa chạy nhanh hơn.
    query_total_mcap_latest = f"""
    WITH MarketCapLatestRanked AS (
        SELECT 
            fid.CompanyID,
            ROUND(ISNULL(fid.MarketCap,0)/{unit}, 2) AS VonHoaMoiNhat,
            ROW_NUMBER() OVER (
                PARTITION BY fid.CompanyID 
                ORDER BY fid.TradingDate DESC
            ) AS Rank
        FROM VSTDataFeed.dbo.FinanceIndexDaily fid WITH (NOLOCK)
        INNER JOIN VSTDataFeed.dbo.Company c ON c.CompanyID = fid.CompanyID
        WHERE 
            ISNULL(fid.MarketCap,0) > 0
            AND c.Status = 1 
            AND c.CatID IN (1,2,5)
    ),
    MarketCapLatest AS (
        SELECT *
        FROM MarketCapLatestRanked
        WHERE Rank = 1
    )
    SELECT SUM(VonHoaMoiNhat) AS TotalMarketCap
    FROM MarketCapLatest;
    """
    total_marketcap = execute_sql_func(query_total_mcap_latest)['TotalMarketCap'].iloc[0]
    return total_listed, total_marketcap


def calculate_market_summary(df_merged, execute_sql_func, current_year=2025, current_quarter='Q3'):
    
    # 1. Lấy tăng trưởng thị trường từ hàm đã có
    print("\nCalculating market summary (Optimized)...")
    print("   Step 1: Re-using 'calculate_industry_growth_rates_abs_base' for market totals...")
    q_int = int(current_quarter[1:])
    try:
        df_market_summary_full = calculate_industry_growth_rates_abs_base(
            df_merged,
            industry_col='MaCoPhieu', # Dùng cột bất kỳ, vì chúng ta chỉ lấy 'Toàn thị trường'
            filter_year=current_year,
            filter_quarter=q_int
        )
        df_market_summary = df_market_summary_full[
            df_market_summary_full['Phân loại'] == 'Toàn thị trường'
        ].copy()
        if df_market_summary.empty:
            print("   ⚠️ Could not calculate market summary.")
            return pd.DataFrame(), pd.DataFrame()
        print("   ✅ Market growth (QoQ, YoY, YTD) calculated.")
    except Exception as e:
        print(f"   ❌ Error calculating market summary: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 2. Tính các chỉ số thống kê (phần riêng của hàm này)
    print("   Step 2: Calculating market coverage statistics...")
    try:
        total_listed, total_marketcap = get_market_totals_latest(execute_sql_func)
        
        df_q_curr = df_merged[
            (df_merged['Nam'] == current_year) & 
            (df_merged['Quy'] == current_quarter)
        ]
        
        reported_firms = df_q_curr['MaCoPhieu'].unique()
        total_reported = len(reported_firms)
        pct_reported = total_reported / total_listed * 100 if total_listed else np.nan

        # Lấy vốn hóa lớn nhất (mới nhất) của các công ty đã báo cáo
        mcap_reported = df_merged[df_merged['MaCoPhieu'].isin(reported_firms)] \
            .groupby('MaCoPhieu')['VonHoa'].max().sum()
            
        pct_mcap_reported = (mcap_reported / total_marketcap * 100) if total_marketcap else np.nan

        df_profit_curr = df_q_curr[df_q_curr['LoiNhuanSauThue'] > 0]
        n_profit = df_profit_curr['MaCoPhieu'].nunique()
        
        # Chỉ tính lỗ nếu có dữ liệu lợi nhuận
        df_loss_curr = df_q_curr[
            (df_q_curr['LoiNhuanSauThue'] <= 0) & 
            (df_q_curr['LoiNhuanSauThue'].notna())
        ]
        n_loss = df_loss_curr['MaCoPhieu'].nunique()

        market_stats_df = pd.DataFrame([{
            'Tổng số công ty niêm yết': total_listed,
            'Tổng số công ty đã công bố (Q hiện tại)': total_reported,
            'Tỷ lệ công bố (%)': f"{pct_reported:.2f}%",
            'Tổng vốn hóa toàn thị trường': total_marketcap,
            'Tổng vốn hóa đã công bố': mcap_reported,
            'Tỷ lệ vốn hóa đã công bố (%)': f"{pct_mcap_reported:.2f}%" if not np.isnan(pct_mcap_reported) else np.nan,
            'Số công ty lãi (Q hiện tại)': n_profit,
            'Số công ty lỗ (Q hiện tại)': n_loss
        }])
        print("   ✅ Market coverage statistics calculated.")
        return df_market_summary, market_stats_df
    except Exception as e:
        print(f"   ❌ Error calculating market stats: {e}")
        return df_market_summary, pd.DataFrame()

def get_financial_reports_filtered_by_list(
    sql_connection_string: str,
    stock_codes_list: List[str],
    term_type: str = "Q",
    report_type: Optional[str] = None, 
    component_type: Optional[str] = None,
    lookback_periods: int = 8
) -> pd.DataFrame:
    """
    ✅ V2.7 (Optimized) - Lấy dữ liệu BCTC theo danh sách.
    - Sửa lỗi logic ưu tiên (HN/ĐL) và lookback_periods.
    - Đảm bảo lấy đúng N kỳ gần nhất theo ưu tiên.
    """

    if not stock_codes_list:
        print("⚠️ Danh sách mã cổ phiếu rỗng. Bỏ qua truy vấn chi tiết.")
        return pd.DataFrame()

    term_type_filter = 1 if term_type == "Y" else 2
    codes_string = ", ".join([f"'{code}'" for code in stock_codes_list])
    stock_code_filter_sql = f"AND ci.CompanyCode IN ({codes_string})"
    sql_params = [report_type, component_type, lookback_periods, term_type_filter]

    sql_query = f"""
    DECLARE @ReportType NVARCHAR(10) = ?;
    DECLARE @ComponentType NVARCHAR(20) = ?;
    DECLARE @LookbackPeriods INT = ?;
    DECLARE @TermTypeFilter INT = ?;
    DECLARE @Unit BIGINT = 1000000;

    ;WITH CompanyInfo AS (
        SELECT CompanyID, CompanyCode, CompanyType, IndustryID
        FROM VSTDataFeed.dbo.Company ci WITH (NOLOCK)
        WHERE ci.Status = 1 AND ci.CatID IN (1, 2, 5)
        {stock_code_filter_sql}
    ),

    PrioritizedReports AS (
        SELECT 
            rd.ReportDataID, rd.CompanyID, rd.YearPeriod, rd.ReportTermID, rd.IsUnited,
            dt.ReportTermTypeID, dt.DisplayOrdering, dt.TermCode,
            rd.AuditStatusID, rd.IsAdjusted, rd.ReportDate, rd.LastUpdate,
            ROW_NUMBER() OVER (
                PARTITION BY rd.CompanyID, rd.YearPeriod, dt.ReportTermID
                ORDER BY 
                    -- 1. Ưu tiên theo lựa chọn của user (nếu có)
                    CASE 
                        WHEN @ReportType = N'HN' AND rd.IsUnited = 0 THEN 1
                        WHEN @ReportType = N'ĐL' AND rd.IsUnited = 1 THEN 1
                        WHEN @ReportType IS NULL THEN 1 -- Nếu user không chọn, mọi loại đều OK
                        ELSE 2 
                    END ASC,
                    -- 2. Ưu tiên mặc định (HN > ĐL)
                    rd.IsUnited ASC, 
                    -- 3. Ưu tiên báo cáo đã kiểm toán
                    rd.AuditStatusID ASC,
                    -- 4. Ưu tiên báo cáo đã điều chỉnh
                    rd.IsAdjusted DESC,
                    -- 5. Lấy báo cáo mới nhất
                    rd.ReportDataID DESC
            ) AS PriorityRank
        FROM VSTDataFeed.dbo.ReportData rd WITH (NOLOCK)
        INNER JOIN CompanyInfo ci ON ci.CompanyID = rd.CompanyID
        INNER JOIN VSTDataFeed.dbo.ReportTerm dt WITH (NOLOCK)
            ON dt.ReportTermID = rd.ReportTermID
        WHERE rd.IsUnited IN (0,1)
          AND dt.ReportTermTypeID = @TermTypeFilter
    ),

    RankedReportPeriods AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY CompanyID
                ORDER BY YearPeriod DESC, DisplayOrdering DESC
            ) AS PeriodRank -- Xếp hạng các kỳ từ mới nhất (1)
        FROM PrioritizedReports
        WHERE PriorityRank = 1 -- Chỉ lấy báo cáo TỐT NHẤT cho mỗi kỳ
    ),

    ReportContext AS (
        SELECT 
            rrp.ReportDataID, rrp.CompanyID, ci.CompanyCode, ci.CompanyType, ci.IndustryID,
            rrp.YearPeriod, rrp.ReportTermID, rrp.ReportDate, rrp.LastUpdate,
            rrp.IsUnited, rrp.AuditStatusID, rrp.IsAdjusted,
            rrp.TermCode, rrp.ReportTermTypeID
        FROM RankedReportPeriods rrp
        INNER JOIN CompanyInfo ci ON ci.CompanyID = rrp.CompanyID
        WHERE rrp.PeriodRank <= @LookbackPeriods -- Lấy N kỳ gần nhất
    ),

    MarketCapLatest AS (
        SELECT fid.CompanyID,
               ROUND(ISNULL(fid.MarketCap, 0) / @Unit, 2) AS VonHoaMoiNhat
        FROM (
            SELECT 
                fid.CompanyID, fid.MarketCap, fid.TradingDate,
                ROW_NUMBER() OVER (PARTITION BY fid.CompanyID ORDER BY fid.TradingDate DESC) AS rk
            FROM VSTDataFeed.dbo.FinanceIndexDaily fid WITH (NOLOCK)
            INNER JOIN CompanyInfo ci ON ci.CompanyID = fid.CompanyID
            WHERE ISNULL(fid.MarketCap, 0) > 0
        ) fid
        WHERE fid.rk = 1
    ),

    FinancialData AS (
        -- PHẦN I: Báo cáo tài chính chính (KQKD, CDKT, LCTT)
        SELECT
            rc.CompanyCode AS MaCoPhieu,
            rc.YearPeriod AS NamBaoCao,
            rc.TermCode AS KyBaoCao,
            dt.Description AS TenKyBaoCaoVN,
            rc.LastUpdate AS NgayCongBo,
            rc.IsUnited,
            rc.CompanyType,
            rc.IndustryID,
            CASE rc.IsUnited WHEN 0 THEN N'HN' WHEN 1 THEN N'ĐL' ELSE N'CTM' END AS LoaiHinhBaoCao,
            rct.Name AS LoaiBaoCaoVN,
            rct.Code AS MaLoaiBaoCao,
            CAST(rn.ReportNormID AS BIGINT) AS ReportNormID,
            COALESCE(ns.NormName, rn.Name, n.NormName) AS TenChiTieuVN,
            CASE WHEN @Unit = 1 THEN ISNULL(rdd.Value, 0)
                 ELSE ROUND(ISNULL(rdd.Value, 0) / @Unit, 2)
            END AS GiaTri,
            ISNULL(mcl.VonHoaMoiNhat, 0.00) AS VonHoa
        FROM ReportContext rc
        INNER JOIN VSTDataFeed.dbo.ReportTerm dt ON dt.ReportTermID = rc.ReportTermID
        LEFT JOIN VSTDataFeed.dbo.ReportDataDetails rdd ON rdd.ReportDataID = rc.ReportDataID
        LEFT JOIN VSTDataFeed.dbo.ReportNorm rn ON rn.ReportNormID = rdd.ReportNormID
        LEFT JOIN VSTDataFeed.dbo.Norms n ON n.NormID = rn.NormID
        LEFT JOIN VSTDataFeed.dbo.ReportComponents rcp ON rn.ReportComponentID = rcp.ReportComponentID
        LEFT JOIN VSTDataFeed.dbo.ReportComponentTypes rct ON rcp.ReportComponentTypeID = rct.ReportComponentTypeID
        LEFT JOIN VSTDataFeed.dbo.NormSpecial ns ON (
             (rc.CompanyType = 0 AND rn.ReportNormID = ns.ReportNormID_CP) OR
             (rc.CompanyType = 1 AND rn.ReportNormID = ns.ReportNormID_CK) OR
             (rc.CompanyType = 2 AND rn.ReportNormID = ns.ReportNormID_NH) OR
             (rc.CompanyType = 3 AND rn.ReportNormID = ns.ReportNormID_Q)  OR
             (rc.CompanyType = 4 AND rn.ReportNormID = ns.ReportNormID_BH)
        )
        LEFT JOIN MarketCapLatest mcl ON mcl.CompanyID = rc.CompanyID
        WHERE (@ComponentType IS NULL OR @ComponentType = N'KQKD' AND rct.Code = 'KQ')
           OR (@ComponentType = N'CDKT' AND rct.Code = 'CD')
           OR (@ComponentType = N'LCTT' AND rct.Code = 'LC')

        UNION ALL

        -- PHẦN II: Chỉ số tài chính (CSTC)
        SELECT
            rc.CompanyCode AS MaCoPhieu,
            rc.YearPeriod AS NamBaoCao,
            rc.TermCode AS KyBaoCao,
            dt.Description AS TenKyBaoCaoVN,
            rc.LastUpdate AS NgayCongBo,
            rc.IsUnited,
            rc.CompanyType,
            rc.IndustryID,
            CASE rc.IsUnited WHEN 0 THEN N'HN' WHEN 1 THEN N'ĐL' ELSE N'CTM' END AS LoaiHinhBaoCao,
            fig.FinanceIndexName AS LoaiBaoCaoVN,
            'CSTC' AS MaLoaiBaoCao,
            CAST(fi.FinanceIndexID AS BIGINT) AS ReportNormID,
            fi.FinanceIndexName AS TenChiTieuVN,
            ROUND(ISNULL(fid.IndexValue, 0), 4) AS GiaTri,
            ISNULL(mcl.VonHoaMoiNhat, 0.00) AS VonHoa
        FROM ReportContext rc
        INNER JOIN VSTDataFeed.dbo.ReportTerm dt ON dt.ReportTermID = rc.ReportTermID
        LEFT JOIN VSTDataFeed.dbo.FinanceIndexData fid ON fid.CompanyID = rc.CompanyID
             AND fid.YearPeriod = rc.YearPeriod AND fid.ReportTermID = rc.ReportTermID
        LEFT JOIN VSTDataFeed.dbo.FinanceIndex fi ON fi.FinanceIndexID = fid.FinanceIndexID
        LEFT JOIN VSTDataFeed.dbo.FinanceIndexGroup fig ON fig.FinanceIndexGroupID = fi.FinanceIndexGroupID
        LEFT JOIN MarketCapLatest mcl ON mcl.CompanyID = rc.CompanyID
        WHERE (@ComponentType IS NULL OR @ComponentType = 'CSTC')
    )

    SELECT *
    FROM FinancialData
    ORDER BY
        MaCoPhieu ASC, NamBaoCao DESC,
        CASE KyBaoCao WHEN 'Q4' THEN 1 WHEN 'Q3' THEN 2 WHEN 'Q2' THEN 3 WHEN 'Q1' THEN 4 ELSE 0 END ASC,
        MaLoaiBaoCao ASC;
    """
    # --- THỰC THI ---
    try:
        conn = pyodbc.connect(sql_connection_string, fast_executemany=True)
        print(f"🟢 Đang tải dữ liệu cho {len(stock_codes_list)} mã (V2.7 Optimized)...")
        df = pd.read_sql(sql_query, conn, params=sql_params)
        conn.close()
        print(f"✅ Thành công: {len(df):,} dòng dữ liệu.")
        return df

    except Exception as e:
        print(f"❌ Lỗi khi thực thi truy vấn SQL: {e}")
        try: conn.close()
        except: pass
        return pd.DataFrame()
# --- FIX: END ---
    
def fetch_data_in_batches(sql_connection_string, stock_codes_to_fetch, batch_size=50):
    """
    Tải dữ liệu chi tiết từ SQL theo các lô nhỏ để tránh lỗi bộ nhớ (MemoryError).
    Sử dụng hàm get_financial_reports_filtered_by_list (V2.7 Optimized)
    """
    all_data = []
    total_codes = len(stock_codes_to_fetch)
    num_batches = (total_codes + batch_size - 1) // batch_size # Tính tổng số lô

    print(f"Bắt đầu tải dữ liệu theo {num_batches} lô (batch size: {batch_size})...")

    for i in range(0, total_codes, batch_size):
        batch_codes = stock_codes_to_fetch[i:i + batch_size]
        
        # Chỉ hiển thị thông báo tiến trình cho lô hiện tại
        print(f"   -> Đang tải lô {i//batch_size + 1}/{num_batches}: {len(batch_codes)} mã cổ phiếu...")
        
        try:
            # Gọi hàm truy vấn SQL cho lô hiện tại (hàm đã tối ưu)
            df_batch = get_financial_reports_filtered_by_list(
                sql_connection_string=sql_connection_string,
                stock_codes_list=batch_codes,
                term_type="Q",
                report_type=None, # Ưu tiên HN, fallback ĐL
                component_type=None, # Lấy tất cả (KQKD, CDKT, LCTT, CSTC)
                lookback_periods=16 # Lấy 16 quý gần nhất (4 năm)
            )
            
            if not df_batch.empty:
                all_data.append(df_batch)
                # Đã có print trong hàm con
                # print(f" 	  ✅ Tải thành công {len(df_batch):,} dòng dữ liệu.")
            else:
                print(" 	  ⚠️ Lô này tải về rỗng.")
        except Exception as e:
            print(f" 	  ❌ Lỗi nghiêm trọng khi tải lô: {e}")
            # Tiếp tục sang lô tiếp theo nếu một lô thất bại

    if all_data:
        # Nối tất cả các DataFrame lô lại thành một DataFrame duy nhất
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()
    
# --- TRONG utils_optimized.py ---
# (Hãy chắc chắn bạn đã có "import streamlit as st" ở đầu file)

def generate_professional_growth_chart_v5(
    df_merged: pd.DataFrame,
    metric_to_plot: str,
    select_year: int,
    select_quarter: str,
    lookback_periods: int = 12,
    cols_to_aggregate: List[str] = ['DoanhThuThuan', 'LoiNhuanTruocThue', 'LoiNhuanSauThue'],
    report_font: str = "Arial",
    base_font_size: int = 11,
    title_font_size_multiplier: float = 1.6,
    axis_label_font_size_multiplier: float = 1.1,
    market_line_color: str = 'rgba(0,0,0,0.9)', 
    market_line_width: float = 3.5,
    market_line_dash: str = 'dash',
    other_line_width: float = 2.0,
    color_palette_name: str = 'Plotly',
    show_range_slider: bool = False,
    add_source_note: Optional[str] = "Nguồn: VSTDataFeed / Tính toán riêng"
) -> Optional[go.Figure]:
    
    print(f"\n--- [V5.4 Final Axis Fix] Starting V5 cho: {metric_to_plot} ---")
    
    # --- Bước 1 & 2: Tính toán (Giữ nguyên) ---
    agg_growth_cols = [f'{col}_Agg_YoY_Growth_Abs' for col in cols_to_aggregate]
    plot_col_name = f'{metric_to_plot}_Agg_YoY_Growth_Abs'
    
    if metric_to_plot not in cols_to_aggregate:
        print(f"   [V5.4] ⚠️ Error: '{metric_to_plot}' not in cols_to_aggregate.")
        return None
    try:
        group_sum_nhom = df_merged.groupby(['NhomPhanTich', 'Nam', 'Quy'])[cols_to_aggregate].sum().reset_index()
        market_sum = df_merged.groupby(['Nam', 'Quy'])[cols_to_aggregate].sum().reset_index()
        market_sum['NhomPhanTich'] = 'Toàn thị trường'
        df_combined_sum = pd.concat([group_sum_nhom, market_sum], ignore_index=True)
        
        quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        df_combined_sum['QuarterNum'] = df_combined_sum['Quy'].map(quarter_map)
        df_combined_sum_sorted = df_combined_sum.sort_values(by=['NhomPhanTich', 'Nam', 'QuarterNum'])
        
        for i, col in enumerate(cols_to_aggregate):
            growth_col_name = agg_growth_cols[i]
            previous_value = df_combined_sum_sorted.groupby('NhomPhanTich')[col].shift(periods=4)
            current_value = df_combined_sum_sorted[col]
            denominator = np.abs(previous_value)
            with np.errstate(divide='ignore', invalid='ignore'):
                df_combined_sum_sorted[growth_col_name] = np.divide(
                    current_value - previous_value, denominator,
                    out=np.full_like(current_value, np.nan, dtype=np.float64),
                    where=(denominator!=0) & (~np.isnan(denominator))
                )
        df_agg_growth_summary = df_combined_sum_sorted[['NhomPhanTich', 'Nam', 'Quy'] + agg_growth_cols].reset_index(drop=True)
        print("   [V5.4] Bước 1&2 (Tính toán) thành công.")
    except Exception as e:
        print(f"   [V5.4] ❌ Error in growth calculation: {e}")
        return None

    # --- Bước 3: Chuẩn bị dữ liệu (Lọc theo kỳ) ---
    print(f"   [V5.4] Step 3: Preparing plot data... Ending at {select_quarter} {select_year} for {lookback_periods} periods.")
    try:
        df_agg_growth_summary['TimeStr'] = df_agg_growth_summary['Nam'].astype(str) + '-' + df_agg_growth_summary['Quy']
        df_agg_growth_summary['TimePeriod'] = pd.PeriodIndex(df_agg_growth_summary['TimeStr'], freq='Q').to_timestamp(how='end')
        df_plot_agg = df_agg_growth_summary.sort_values(by=['NhomPhanTich', 'TimePeriod'])
        
        all_periods_ts_df = df_plot_agg[(df_plot_agg['NhomPhanTich'] == 'Toàn thị trường')]
        if all_periods_ts_df.empty:
            print("   [V5.4] ❌ LỖI: Không tìm thấy 'Toàn thị trường' trong dữ liệu.")
            return None
            
        all_periods_ts = pd.to_datetime(all_periods_ts_df['TimePeriod'].sort_values().unique())
        
        target_period_str = f"{select_year}-{select_quarter}"
        target_ts = pd.Period(target_period_str, freq='Q').to_timestamp(how='end')
        
        all_periods_ts_filtered = all_periods_ts[all_periods_ts <= target_ts]

        if len(all_periods_ts_filtered) == 0:
            print(f"   [V5.4] ⚠️ Không tìm thấy dữ liệu 'Toàn thị trường' cho kỳ {select_quarter} {select_year} hoặc sớm hơn.")
            periods_to_plot = all_periods_ts[-lookback_periods:]
        else:
            periods_to_plot = all_periods_ts_filtered[-lookback_periods:]
        
        # =======================================================
        # --- SỬA LỖI TRỤC X (TẠO BIẾN MỚI) ---
        # Đây là danh sách các vạch chia (ticks) chúng ta THỰC SỰ muốn hiển thị
        axis_tickvals = periods_to_plot
        # Và đây là nhãn cho các vạch chia đó
        axis_ticktext = [f"{t.year}\n{t.to_period('Q').strftime('Q%q')}" for t in pd.to_datetime(axis_tickvals)]
        # =======================================================
        
        df_plot_agg_filtered = df_plot_agg[df_plot_agg['TimePeriod'].isin(periods_to_plot)].copy()

        if df_plot_agg_filtered.empty:
            print("   [V5.4] ⚠️ LỖI: df_plot_agg_filtered BỊ RỖNG sau khi lọc (lỗi .isin).")
            return None
        
        # SỬA LỖI return None (thay dropna bằng fillna)
        df_plot_agg_filtered[plot_col_name] = df_plot_agg_filtered[plot_col_name].fillna(0)
        print(f"   [V5.4] Đã fillna(0) cho cột {plot_col_name}.")
            
    except Exception as e:
        print(f"   [V5.4] ❌ Error preparing plot data (Step 3): {e}")
        return None

    # --- Bước 4: Vẽ biểu đồ (Phong cách Tối giản) ---
    print(f"   [V5.4] Step 4: Plotting V5 (Minimalist) chart for '{metric_to_plot}'...")
    try:
        metric_title = metric_to_plot.replace('DoanhThu', 'Doanh Thu ').replace('LoiNhuan', 'Lợi Nhuận ').replace('TruocThue', 'Trước Thuế ').replace('SauThue', 'Sau Thuế ')
        start_ts = df_plot_agg_filtered['TimePeriod'].min()
        end_ts = df_plot_agg_filtered['TimePeriod'].max()
        start_period_label = pd.Timestamp(start_ts).to_period('Q')
        end_period_label = pd.Timestamp(end_ts).to_period('Q')
        
        try:
            color_sequence = getattr(px.colors.qualitative, color_palette_name)
        except AttributeError:
            color_sequence = px.colors.qualitative.Plotly

        fig = px.line(
            df_plot_agg_filtered, x='TimePeriod', y=plot_col_name, color='NhomPhanTich',
            color_discrete_sequence=color_sequence, 
            markers=True,          
            #line_shape='linear',   # Giữ 'linear' (thẳng)
            line_shape='spline',# <-- Đổi 'linear' TRỞ LẠI 'spline'
            title=f'<b>Tăng trưởng Tổng {metric_title} YoY theo Nhóm Phân tích</b><br><sup><i>Phương pháp: (Hiện tại - Trước) / |Trước|, giai đoạn {start_period_label}-{end_period_label}</i></sup>',
            labels={'TimePeriod': '', plot_col_name: 'Tăng trưởng YoY (%)', 'NhomPhanTich': ''},
            template='plotly_white'
        )
        
        fig.update_layout(
            font=dict(family=report_font, size=base_font_size, color='black'),
            title=dict(font_size=base_font_size * title_font_size_multiplier, x=0.05, xanchor='left'),
            xaxis_title=None,
            yaxis_title=dict(font_size=base_font_size * axis_label_font_size_multiplier, standoff=10),
            yaxis_tickformat='.0%', 
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font_size=base_font_size, bgcolor='rgba(255,255,255,0.6)'),
            margin=dict(l=60, r=30, t=110, b=80),
            plot_bgcolor='white', paper_bgcolor='white',
            annotations=[
                dict(xref='paper', yref='paper', x=0, y=-0.2, showarrow=False,
                     text=add_source_note, font=dict(size=base_font_size * 0.9, color='grey'), align='left')
            ] if add_source_note else [],
            hoverlabel=dict(
                bgcolor="white",
                font_size=base_font_size,
                font_family=report_font
            )
        )
        
        # =======================================================
        # --- SỬA LỖI TRỤC X HIỂN THỊ Q4 2025 ---
        fig.update_xaxes(
            # Xóa dtick và tickformat
            # dtick="M3", 
            # tickformat="%Y\n%q", 
            
            # Thêm tickvals và ticktext
            tickvals=axis_tickvals, # <-- SỬA LỖI
            ticktext=axis_ticktext, # <-- SỬA LỖI
            
            tickfont_size=base_font_size,
            showgrid=False, showline=False,
            rangeslider_visible=show_range_slider,
            showspikes=True, spikemode='across', spikedash='dot', spikethickness=1
            # Xóa range (để tickvals kiểm soát)
            # range=[start_ts, end_ts] 
        )
        # =======================================================
        
        fig.update_yaxes(
            tickfont_size=base_font_size, 
            showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.05)',
            showline=False, 
            zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.8)',
            showspikes=True, spikemode='across', spikedash='dot', spikethickness=1
        )
        
        def apply_trace_styling(trace):
            trace_color_rgb = trace.marker.color
            if trace_color_rgb and trace_color_rgb.startswith('rgb'):
                fill_color_rgba = trace_color_rgb.replace('rgb', 'rgba').replace(')', ', 0.1)')
            else:
                fill_color_rgba = 'rgba(128,128,128,0.1)'
            
            if trace.name == 'Toàn thị trường':
                trace.update(
                    line=dict(width=market_line_width, color=market_line_color, dash=market_line_dash),
                    hovertemplate='<b>Toàn thị trường</b><br>%{x|%YQ%q}: %{y:.1%}<extra></extra>',
                    fill='tozeroy',
                    fillcolor='rgba(0,0,0,0.15)'
                )
            else:
                trace.update(
                    line=dict(width=other_line_width),
                    hovertemplate='<b>'+trace.name+'</b><br>%{x|%YQ%q}: %{y:.1%}<extra></extra>',
                    fill='tozeroy',
                    fillcolor=fill_color_rgba
                )

        fig.for_each_trace(apply_trace_styling)
        
        print(f"   [V5.4] ✅ V5 (Minimalist+Filter) Chart plotting completed.")
        return fig
    except Exception as e:
        print(f"   [V5.4] ❌ Error in plotting V5 chart (Step 4): {e}")
        return None
