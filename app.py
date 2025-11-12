import os
import sys
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import sqlite3
import akshare as ak
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import numpy as np
import io
import base64

app = Flask(__name__)


# 配置
class Config:
    # 在 Render 上使用绝对路径
    DB_PATH = os.path.join(os.getcwd(), 'stock_data.db')
    DATA_DIR = os.path.join(os.getcwd(), 'stock_data')
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')


app.config.from_object(Config)


class StockDataPlatform:
    """股票数据管理平台 - 修复版本"""

    def __init__(self, db_path=None, data_dir=None):
        self.db_path = db_path or app.config['DB_PATH']
        self.data_dir = data_dir or app.config['DATA_DIR']
        self.setup_directories()
        self.setup_database()

    def setup_directories(self):
        """创建数据目录"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "reports"), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "charts"), exist_ok=True)
            print(f"目录创建成功: {self.data_dir}")
        except Exception as e:
            print(f"目录创建失败: {e}")

    def setup_database(self):
        """初始化数据库 - 修复 SQLite 语法"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 修复：使用 AUTOINCREMENT 而不是 AUTO_INCREMENT
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS limit_up_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT,
                    limit_reason TEXT,
                    limit_price REAL,
                    increase_rate REAL,
                    turnover_rate REAL,
                    封单额 REAL,
                    流通市值 REAL,
                    first_limit_time TEXT,
                    last_limit_time TEXT,
                    open_times INTEGER,
                    continuous_boards INTEGER,
                    limit_analysis TEXT,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, code)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS smash_coefficient_results (
                    date TEXT PRIMARY KEY,
                    smash_coefficient REAL,
                    max_continuous_boards INTEGER,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
            print("数据库初始化成功")
        except Exception as e:
            print(f"数据库初始化失败: {e}")

    def update_limit_up_data(self, date_str=None):
        """更新单日涨停数据"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        try:
            df = ak.stock_zt_pool_em(date=date_str)

            if df.empty:
                return {"status": "no_data", "message": f"{date_str} 无涨停数据"}

            df_cleaned = self.clean_limit_up_data(df, date_str)

            conn = sqlite3.connect(self.db_path)
            records_count = 0

            for _, row in df_cleaned.iterrows():
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO limit_up_data 
                        (date, code, name, limit_reason, limit_price, increase_rate, 
                         turnover_rate, 封单额, 流通市值, first_limit_time, last_limit_time,
                         open_times, continuous_boards, limit_analysis)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['日期'], row['代码'], row['名称'], row['涨停原因'],
                        row['涨停价'], row['涨幅'], row['换手率'], row['封单额'],
                        row['流通市值'], row['首封'], row['封住'], row['开板'],
                        row['几天几板'], row['涨停分析']
                    ))
                    records_count += 1
                except Exception as e:
                    print(f"插入数据失败: {e}")

            conn.commit()
            conn.close()

            return {"status": "success", "message": f"成功更新 {records_count} 条涨停数据"}

        except Exception as e:
            return {"status": "error", "message": f"更新涨停数据失败: {e}"}

    def update_limit_up_data_by_period(self, start_date, end_date):
        """按期间更新涨停数据"""
        try:
            # 检查日期格式，如果是YYYY-MM-DD格式，转换为YYYYMMDD
            if '-' in start_date:
                start_date = start_date.replace('-', '')
            if '-' in end_date:
                end_date = end_date.replace('-', '')

            # 转换日期格式
            start = datetime.strptime(start_date, "%Y%m%d")
            end = datetime.strptime(end_date, "%Y%m%d")

            if start > end:
                return {"status": "error", "message": "开始日期不能晚于结束日期"}

            # 计算日期范围
            date_range = []
            current_date = start
            while current_date <= end:
                date_range.append(current_date.strftime("%Y%m%d"))
                current_date += timedelta(days=1)

            # 更新数据
            success_count = 0
            no_data_count = 0
            error_count = 0

            for date_str in date_range:
                result = self.update_limit_up_data(date_str)
                if result["status"] == "success":
                    success_count += 1
                elif result["status"] == "no_data":
                    no_data_count += 1
                else:
                    error_count += 1

            return {
                "status": "success",
                "message": f"期间更新完成！成功更新: {success_count} 天，无数据: {no_data_count} 天，更新失败: {error_count} 天"
            }

        except Exception as e:
            return {"status": "error", "message": f"期间更新失败: {e}"}

    def clean_limit_up_data(self, df, date_str):
        """清洗涨停数据"""
        column_mapping = {
            '代码': '代码', '名称': '名称', '涨跌幅': '涨幅', '最新价': '涨停价',
            '换手率': '换手率', '封板资金': '封单额', '流通市值': '流通市值',
            '首次封板时间': '首封', '最后封板时间': '封住', '炸板次数': '开板',
            '连板数': '几天几板', '涨停原因': '涨停原因', '涨停分析': '涨停分析'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        df['日期'] = date_formatted

        required_columns = [
            '日期', '代码', '名称', '涨停原因', '涨停价', '涨幅',
            '换手率', '封单额', '流通市值', '首封', '封住', '开板',
            '几天几板', '涨停分析'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        df['代码'] = df['代码'].astype(str).apply(lambda x: x.zfill(6))

        return df[required_columns]

    def get_limit_up_data(self, start_date=None, end_date=None):
        """获取涨停数据"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = "SELECT * FROM limit_up_data WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date DESC, code"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            print(f"获取涨停数据失败: {e}")
            return pd.DataFrame()

    def calculate_smash_coefficient(self, start_date=None, end_date=None):
        """计算砸盘系数"""
        try:
            data = self.get_limit_up_data(start_date, end_date)

            if data.empty:
                return {"status": "error", "message": "没有数据可用于计算砸盘系数"}

            def extract_boards(text):
                if pd.isna(text):
                    return 1
                text = str(text).strip()

                if '首板' in text:
                    return 1

                if '天' in text and '板' in text:
                    import re
                    nums = re.findall(r'\d+', text)
                    if nums:
                        return int(nums[0])

                try:
                    return int(text)
                except:
                    pass

                return 1

            data['连板数'] = data['continuous_boards'].apply(extract_boards)

            dates = sorted(data['date'].unique())
            daily_stats = {}

            for date in dates:
                day_data = data[data['date'] == date]
                counts = {}
                for i in range(1, 17):
                    count = len(day_data[day_data['连板数'] == i])
                    counts[i] = count
                daily_stats[date] = counts

            results = []
            for i in range(1, len(dates)):
                current_date = dates[i]
                prev_date = dates[i - 1]

                if prev_date not in daily_stats or current_date not in daily_stats:
                    continue

                curr = daily_stats[current_date]
                prev = daily_stats[prev_date]

                ratio_sum = 0
                valid_ratios = 0

                for n in range(3, 17):
                    prev_n_minus_1 = prev.get(n - 1, 0)
                    curr_n = curr.get(n, 0)

                    if prev_n_minus_1 > 0:
                        ratio = curr_n / prev_n_minus_1
                        ratio_sum += ratio
                        valid_ratios += 1

                if valid_ratios > 0:
                    smash_coef = round((ratio_sum / 4) * 10, 2)
                else:
                    smash_coef = 0

                max_board = max(k for k, v in curr.items() if v > 0) if any(v > 0 for v in curr.values()) else 0

                results.append({
                    'date': current_date,
                    'smash_coefficient': smash_coef,
                    'max_continuous_boards': max_board
                })

                self.save_smash_coefficient(current_date, smash_coef, max_board)

            return {
                "status": "success",
                "data": results,
                "message": f"计算完成，共 {len(results)} 条记录"
            }

        except Exception as e:
            return {"status": "error", "message": f"计算砸盘系数时出错: {e}"}

    def save_smash_coefficient(self, date, coefficient, max_boards):
        """保存砸盘系数到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO smash_coefficient_results 
                (date, smash_coefficient, max_continuous_boards)
                VALUES (?, ?, ?)
            ''', (date, coefficient, max_boards))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"保存砸盘系数失败: {e}")

    def get_smash_coefficient_history(self, start_date=None, end_date=None):
        """获取历史砸盘系数"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM smash_coefficient_results WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            return {
                "status": "success",
                "data": df.to_dict('records')
            }
        except Exception as e:
            return {"status": "error", "message": f"获取砸盘系数历史失败: {e}"}

    def generate_smash_earn_chart(self, start_date=None, end_date=None):
        """生成砸盘系数与挣钱效应的折线图，返回base64图片"""
        try:
            result = self.get_smash_coefficient_history(start_date, end_date)

            if result["status"] != "success" or not result["data"]:
                return {"status": "error", "message": "没有数据可生成图表"}

            data = pd.DataFrame(result["data"])
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')

            # 创建图表
            fig, ax1 = plt.subplots(figsize=(12, 6))

            plt.title('市场情绪分析-砸盘系数vs挣钱效应', fontsize=14, fontweight='bold', pad=20)

            # 绘制砸盘系数
            color1 = 'red'
            ax1.set_xlabel('日期')
            ax1.set_ylabel('砸盘系数', color=color1)
            smash_data = data['smash_coefficient'].clip(lower=0)
            line1 = ax1.plot(data['date'], smash_data,
                             color=color1, marker='o', markersize=6, linewidth=2.5, label='砸盘系数')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            # 创建第二个Y轴
            ax2 = ax1.twinx()
            color2 = 'blue'
            ax2.set_ylabel('挣钱效应（最高连板数）', color=color2)
            earn_data = data['max_continuous_boards'].clip(lower=0)
            line2 = ax2.plot(data['date'], earn_data,
                             color=color2, marker='s', markersize=6, linewidth=2.5, label='挣钱效应')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(0, max(earn_data.max() + 2, 10))

            # 在砸盘系数折线上标注数值
            for i, (date, value) in enumerate(zip(data['date'], smash_data)):
                ax1.annotate(f'{value:.2f}',
                             (date, value),
                             textcoords="offset points",
                             xytext=(0, 12),
                             ha='center',
                             fontsize=9,
                             fontweight='bold',
                             color=color1,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

            # 在挣钱效应折线上标注数值
            for i, (date, value) in enumerate(zip(data['date'], earn_data)):
                ax2.annotate(f'{int(value)}',
                             (date, value),
                             textcoords="offset points",
                             xytext=(0, -15),
                             ha='center',
                             fontsize=9,
                             fontweight='bold',
                             color=color2,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            plt.gcf().autofmt_xdate()

            # 转换为base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)

            return {"status": "success", "image": plot_url}

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"生成折线图时出错: {e}")
            return {"status": "error", "message": f"生成折线图时出错: {e}"}

    def generate_board_matrix_chart(self, start_date=None, end_date=None):
        """生成连板公司分布矩阵图"""
        try:
            # 获取涨停数据
            limit_up_data = self.get_limit_up_data(start_date, end_date)

            if limit_up_data.empty:
                return {"status": "error", "message": "没有涨停数据可生成矩阵图"}

            # 提取连板数
            def extract_boards(text):
                if pd.isna(text):
                    return 1
                text = str(text).strip()
                if '首板' in text:
                    return 1
                if '天' in text and '板' in text:
                    import re
                    nums = re.findall(r'\d+', text)
                    if nums:
                        return int(nums[0])
                try:
                    return int(text)
                except:
                    pass
                return 1

            limit_up_data['连板数'] = limit_up_data['continuous_boards'].apply(extract_boards)

            # 筛选连板数>=2的数据
            multi_board_data = limit_up_data[limit_up_data['连板数'] >= 2]

            if multi_board_data.empty:
                # 如果没有连板数据，创建一个空的图表
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, '无连板数据', ha='center', va='center', transform=ax.transAxes, fontsize=16)
                ax.set_title('连板公司分布矩阵', fontsize=14, fontweight='bold')
                ax.axis('off')
            else:
                # 按日期和连板数分组，获取公司名称
                grouped = multi_board_data.groupby(['date', '连板数'])['name'].apply(list).reset_index()

                # 获取所有日期和连板数
                all_dates = sorted(multi_board_data['date'].unique())
                all_boards = sorted(multi_board_data['连板数'].unique())

                # 创建矩阵数据
                matrix_data = []
                row_labels = []

                for board in all_boards:
                    row_data = []
                    for date in all_dates:
                        # 获取该日期该连板数的公司列表
                        companies = grouped[(grouped['date'] == date) & (grouped['连板数'] == board)]['name']
                        if not companies.empty:
                            company_list = companies.iloc[0]
                            # 显示公司名称，限制显示数量
                            if len(company_list) > 3:
                                company_text = f"{len(company_list)}家公司"
                            else:
                                company_text = ', '.join(company_list)
                            row_data.append(company_text)
                        else:
                            row_data.append("")
                    matrix_data.append(row_data)
                    row_labels.append(f"{board}板")

                # 创建图表 - 优化居中显示
                fig, ax = plt.subplots(figsize=(max(12, len(all_dates) * 1.8), max(8, len(all_boards) * 1.0)))

                # 创建表格 - 优化居中
                table = ax.table(
                    cellText=matrix_data,
                    rowLabels=row_labels,
                    colLabels=[date[5:] for date in all_dates],  # 只显示月-日
                    cellLoc='center',
                    loc='center',
                    bbox=[0.05, 0.05, 0.95, 0.9]  # 调整bbox使表格居中
                )

                # 设置表格样式
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)

                # 设置单元格颜色
                for i in range(len(all_boards)):
                    for j in range(len(all_dates)):
                        if i == 0 or j == 0:
                            try:
                                table[(i, j)].set_facecolor('#f0f0f0')
                                table[(i, j)].set_text_props(weight='bold')
                            except KeyError:
                                continue
                        elif matrix_data[i][j]:
                            try:
                                table[(i + 1, j + 1)].set_facecolor('#e6f3ff')
                            except KeyError:
                                continue

                # 设置标题
                ax.set_title('连板公司分布矩阵', fontsize=14, fontweight='bold', pad=20)
                ax.axis('off')

            plt.tight_layout()

            # 转换为base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)

            return {"status": "success", "image": plot_url}

        except Exception as e:
            print(f"生成矩阵图时出错: {e}")
            return {"status": "error", "message": f"生成矩阵图时出错: {e}"}

    def get_board_matrix_details(self, start_date=None, end_date=None):
        """获取连板矩阵的详细数据"""
        try:
            # 获取涨停数据
            limit_up_data = self.get_limit_up_data(start_date, end_date)

            if limit_up_data.empty:
                return {"status": "error", "message": "没有涨停数据"}

            # 提取连板数
            def extract_boards(text):
                if pd.isna(text):
                    return 1
                text = str(text).strip()
                if '首板' in text:
                    return 1
                if '天' in text and '板' in text:
                    import re
                    nums = re.findall(r'\d+', text)
                    if nums:
                        return int(nums[0])
                try:
                    return int(text)
                except:
                    pass
                return 1

            limit_up_data['连板数'] = limit_up_data['continuous_boards'].apply(extract_boards)

            # 筛选连板数>=2的数据
            multi_board_data = limit_up_data[limit_up_data['连板数'] >= 2]

            if multi_board_data.empty:
                return {"status": "success", "data": {}}

            # 修复分组警告
            grouped = multi_board_data.groupby(['date', '连板数']).apply(
                lambda x: x[['name', 'code', 'limit_reason']].to_dict('records'),
                include_groups=False
            ).to_dict()

            # 构建详细数据
            matrix_details = {}
            for (date, board), stocks in grouped.items():
                if date not in matrix_details:
                    matrix_details[date] = {}
                matrix_details[date][int(board)] = stocks

            return {
                "status": "success",
                "data": matrix_details
            }

        except Exception as e:
            return {"status": "error", "message": f"获取矩阵详情时出错: {e}"}

    def get_platform_status(self):
        """获取平台状态"""
        try:
            conn = sqlite3.connect(self.db_path)
            status = {}

            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM limit_up_data")
            status['涨停数据总量'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT date) FROM limit_up_data")
            status['交易日期数'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT code) FROM limit_up_data")
            status['涉及股票数'] = cursor.fetchone()[0]

            cursor.execute("SELECT MAX(date) FROM limit_up_data")
            status['最新数据日期'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM smash_coefficient_results")
            status['砸盘系数数据量'] = cursor.fetchone()[0]

            conn.close()

            return {"status": "success", "data": status}
        except Exception as e:
            return {"status": "error", "message": f"获取平台状态失败: {e}"}


# 创建平台实例
platform = StockDataPlatform()


# 路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify(platform.get_platform_status())


@app.route('/api/update_data', methods=['POST'])
def api_update_data():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    if not start_date or not end_date:
        return jsonify({"status": "error", "message": "请提供开始日期和结束日期"})

    return jsonify(platform.update_limit_up_data_by_period(start_date, end_date))


@app.route('/api/calculate_smash', methods=['POST'])
def api_calculate_smash():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    return jsonify(platform.calculate_smash_coefficient(start_date, end_date))


@app.route('/api/history')
def api_history():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return jsonify(platform.get_smash_coefficient_history(start_date, end_date))


@app.route('/api/line_chart')
def api_line_chart():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return jsonify(platform.generate_smash_earn_chart(start_date, end_date))


@app.route('/api/matrix_chart')
def api_matrix_chart():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return jsonify(platform.generate_board_matrix_chart(start_date, end_date))


@app.route('/api/matrix_details')
def api_matrix_details():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return jsonify(platform.get_board_matrix_details(start_date, end_date))


@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "message": "服务运行正常"})


if __name__ == '__main__':
    # 启动应用
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)