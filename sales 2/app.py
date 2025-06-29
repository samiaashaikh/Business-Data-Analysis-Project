from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask import send_file
import pandas as pd
import os
import json
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
import io

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith((".csv", ".xlsx")):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            session["uploaded_file"] = filepath
            flash("File uploaded successfully!", "success")
            return render_template("dashboard.html")
        else:
            flash("Invalid file format! Please upload a CSV or Excel file.", "danger")
    return render_template("upload.html")

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')






@app.route("/sales_analysis", methods=["GET", "POST"])
def sales_analysis():
    file_path = session.get("uploaded_file")
    
    if not file_path:
        flash("No file uploaded! Please upload a file first.", "danger")
        return redirect(url_for("upload_file"))

    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filters
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    category = request.form.get("category")
    location = request.form.get("location")
    payment_method = request.form.get("payment_method")

    if start_date and end_date:
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    if category and category != "All":
        df = df[df["Category"] == category]
    if location and location != "All":
        df = df[df["Store Location"] == location]
    if payment_method and payment_method != "All":
        df = df[df["Payment Method"] == payment_method]

    # Monthly Sales Trends
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    monthly_sales = df.groupby("Month")["Total Sales"].sum().reset_index()
    sales_trend_chart = px.line(monthly_sales, x="Month", y="Total Sales", title="📈 Monthly Sales Trends")

    # Top-Selling Products
    top_products = df.groupby("Product Name")["Quantity Sold"].sum().reset_index().sort_values(by="Quantity Sold", ascending=False).head(10)
    top_products_chart = px.bar(top_products, x="Product Name", y="Quantity Sold", title="🏆 Top-Selling Products")

    # Slow-Moving Products
    slow_products = df.groupby("Product Name")["Quantity Sold"].sum().reset_index().sort_values(by="Quantity Sold", ascending=True).head(10)
    slow_products_chart = px.bar(slow_products, x="Product Name", y="Quantity Sold", title="🐢 Slow-Moving Products")

    # Category-Wise Sales
    category_sales = df.groupby("Category")["Total Sales"].sum().reset_index()
    category_sales_chart = px.pie(category_sales, names="Category", values="Total Sales", title="📊 Category-Wise Sales Distribution")

    # Convert charts to HTML
    sales_trend_html = pio.to_html(sales_trend_chart, full_html=False)
    top_products_html = pio.to_html(top_products_chart, full_html=False)
    slow_products_html = pio.to_html(slow_products_chart, full_html=False)
    category_sales_html = pio.to_html(category_sales_chart, full_html=False)

    return render_template(
        "sales_analysis.html",
        sales_trend_html=sales_trend_html,
        top_products_html=top_products_html,
        slow_products_html=slow_products_html,
        category_sales_html=category_sales_html,
        categories=df["Category"].unique(),
        locations=df["Store Location"].unique(),
        payment_methods=df["Payment Method"].unique()
    )








# 📌 Stock & Inventory Analysis Route

@app.route("/stock-inventory", methods=["GET", "POST"])
def stock_inventory():
    file_path = session.get("uploaded_file")  # Retrieve file path from session

    if not file_path or not os.path.exists(file_path):
        flash("No file uploaded. Please upload a file first.", "warning")
        return redirect(url_for("upload_file"))  # Redirect if no file uploaded

    # Load CSV or Excel File
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Check if required columns exist
    required_columns = {"Product Name", "Category", "Stock", "Unit Price", "Date"}
    if not required_columns.issubset(df.columns):
        flash("Missing required columns in the file.", "danger")
        return redirect(url_for("dashboard"))

    df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime

    # Filters
    category_filter = request.form.get("category")
    date_start = request.form.get("date_start")
    date_end = request.form.get("date_end")
    stock_status = request.form.get("stock_status")

    if date_start and date_end:
        df = df[(df["Date"] >= date_start) & (df["Date"] <= date_end)]

    if category_filter and category_filter != "All":
        df = df[df["Category"] == category_filter]

    # 🟢 Stock Summary
    stock_data = df.groupby(["Product Name", "Category"], as_index=False).agg({
        "Stock": "sum",
        "Unit Price": "mean",
    })
    
    stock_data["Stock Value"] = stock_data["Stock"] * stock_data["Unit Price"]
    
    # Identify Low Stock (Threshold: 10 units)
    stock_data["Low Stock"] = stock_data["Stock"] < 10
    
    # Identify Dead Stock (Example Condition: Stock > 50)
    stock_data["Dead Stock"] = stock_data["Stock"] > 50  

    if stock_status == "low_stock":
        stock_data = stock_data[stock_data["Low Stock"]]
    elif stock_status == "dead_stock":
        stock_data = stock_data[stock_data["Dead Stock"]]

    # 📊 Stock Turnover Rate (Bar Chart)
    stock_turnover_chart = px.bar(
        stock_data, x="Product Name", y="Stock", 
        title="Stock Turnover Rate", color="Category"
    )
    stock_turnover_html = stock_turnover_chart.to_html(full_html=False)

    # 📊 Stock Value (Pie Chart)
    stock_value_chart = px.pie(
        stock_data, names="Product Name", values="Stock Value", 
        title="Stock Value Distribution"
    )
    stock_value_html = stock_value_chart.to_html(full_html=False)

    categories = df["Category"].unique().tolist()
    
    return render_template("stock_inventory.html", 
                           stock_data=stock_data.to_dict(orient="records"),
                           stock_turnover_html=stock_turnover_html,
                           stock_value_html=stock_value_html,
                           categories=categories)


#-----------------Profitability-----------------------------------
@app.route('/profitability-analysis', methods=['GET', 'POST'])
def profitability_analysis():
    if 'uploaded_file' not in session:
        return redirect(url_for('dashboard'))  # Redirect if no file uploaded
    
    filepath = session['uploaded_file']
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    # Calculate Profit (Assuming Cost Price = 70% of Selling Price)
    df["Cost Price"] = df["Unit Price"] * 0.7  
    df["Profit"] = (df["Unit Price"] - df["Cost Price"]) * df["Quantity Sold"]

    # Get Unique Values for Dropdowns
    categories = df["Category"].dropna().unique().tolist()
    locations = df["Store Location"].dropna().unique().tolist()
    payment_modes = df["Payment Method"].dropna().unique().tolist()

    # Apply Filters
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    category = request.form.get("category")
    location = request.form.get("location")
    payment_mode = request.form.get("payment_mode")

    if start_date and end_date:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    if category and category != "All":
        df = df[df["Category"] == category]
    
    if location and location != "All":
        df = df[df["Store Location"] == location]
    
    if payment_mode and payment_mode != "All":
        df = df[df["Payment Method"] == payment_mode]

    # Profit Calculation
    profit_per_product = df.groupby("Product Name")["Profit"].sum().reset_index()
    category_profit = df.groupby("Category")["Profit"].sum().reset_index()
    
    return render_template("profitability_analysis.html", 
                           profit_per_product=profit_per_product.to_dict(orient="records"), 
                           category_profit=category_profit.to_dict(orient="records"),
                           categories=categories,
                           locations=locations,
                           payment_modes=payment_modes)








#---------------------------------------------Customer---------------------------------
@app.route('/customer-patterns', methods=['GET', 'POST'])
def customer_patterns():
    if 'uploaded_file' not in session:
        return redirect(url_for('dashboard'))  # Redirect if no file uploaded

    filepath = session['uploaded_file']
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # 🔹 **Repeat Purchase Analysis**
    repeat_purchases = df.groupby("Product Name")["Customer ID"].nunique().reset_index()
    repeat_purchases.columns = ["Product Name", "Unique Customers"]
    repeat_purchases = repeat_purchases.sort_values(by="Unique Customers", ascending=False)

    # 🔹 **Bundle Recommendations (Frequent Product Pairs)**
    product_pairs = df.groupby("Customer ID")["Product Name"].apply(list).tolist()
    from itertools import combinations
    pair_counts = {}
    for items in product_pairs:
        for combo in combinations(set(items), 2):  # Unique combinations
            pair_counts[combo] = pair_counts.get(combo, 0) + 1
    bundle_recommendations = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # 🔹 **Seasonal Demand Analysis**
    df["Month"] = df["Date"].dt.strftime("%B")
    seasonal_demand = df.groupby("Month")["Product Name"].count().reset_index()
    seasonal_demand.columns = ["Month", "Total Sales"]
    seasonal_demand = seasonal_demand.sort_values(by="Total Sales", ascending=False)

    return render_template("customer_patterns.html", 
                           repeat_purchases=repeat_purchases.to_dict(orient="records"),
                           bundle_recommendations=bundle_recommendations,
                           seasonal_demand=seasonal_demand.to_dict(orient="records"))


#-------------------------Customer------------------------------

@app.route('/supplier-orders', methods=['GET', 'POST'])
def supplier_orders():
    if 'uploaded_file' not in session:
        return redirect(url_for('dashboard'))  # Redirect if no file uploaded

    filepath = session['uploaded_file']
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    # 🔹 Mock supplier data (Replace with real supplier data if available)
    df["Supplier"] = df["Product Name"].apply(lambda x: f"Supplier {ord(x[0]) % 5 + 1}")  # Generate supplier names
    
    # 🔹 **Best & Worst Suppliers**
    supplier_performance = df.groupby("Supplier")["Total Sales"].sum().reset_index()
    supplier_performance.columns = ["Supplier", "Total Sales"]
    supplier_performance = supplier_performance.sort_values(by="Total Sales", ascending=False)

    # 🔹 **Supplier Delivery Performance (Mock Late Delivery Calculation)**
    import random
    df["Late Deliveries"] = df["Supplier"].apply(lambda x: random.randint(0, 5))  # Random late deliveries
    supplier_delays = df.groupby("Supplier")["Late Deliveries"].sum().reset_index()
    supplier_delays.columns = ["Supplier", "Late Deliveries"]
    
    # 🔹 **Optimal Order Quantity (Simple Moving Average for Reordering)**
    df["Order Quantity"] = df["Quantity Sold"].rolling(window=3, min_periods=1).mean()  # Moving average demand
    optimal_orders = df.groupby("Product Name")["Order Quantity"].mean().reset_index()
    optimal_orders.columns = ["Product Name", "Recommended Order Quantity"]

    return render_template("supplier_orders.html", 
                           supplier_performance=supplier_performance.to_dict(orient="records"),
                           supplier_delays=supplier_delays.to_dict(orient="records"),
                           optimal_orders=optimal_orders.to_dict(orient="records"))

#-------------Advance-------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

@app.route('/advanced-analysis', methods=['GET', 'POST'])
def advanced_analysis():
    if 'uploaded_file' not in session:
        return redirect(url_for('dashboard'))  # Redirect if no file uploaded

    filepath = session['uploaded_file']
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    # Ensure required columns exist
    if not {'Date', 'Total Sales', 'Customer ID', 'Quantity Sold'}.issubset(df.columns):
        flash("Invalid file format! Required columns not found.", "danger")
        return redirect(url_for('dashboard'))

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    ## 📈 **1️⃣ Predict Future Sales (Linear Regression)**
    df_monthly = df.groupby("Month")["Total Sales"].sum().reset_index()
    X = df_monthly[["Month"]]
    y = df_monthly["Total Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_sales = model.predict(np.array([[i] for i in range(1, 13)]))  # Predict for 12 months

    ## 📦 **2️⃣ Stock Optimization (Moving Average)**
    df['Stock Needed'] = df['Quantity Sold'].rolling(window=3, min_periods=1).mean()
    stock_recommendations = df.groupby("Product Name")["Stock Needed"].mean().reset_index()
    stock_recommendations.columns = ["Product Name", "Recommended Stock"]

    ## 👥 **3️⃣ Customer Segmentation (K-Means Clustering)**
    customer_df = df.groupby("Customer ID")["Total Sales"].sum().reset_index()
    scaler = StandardScaler()
    customer_df["Total Sales Scaled"] = scaler.fit_transform(customer_df[["Total Sales"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df["Segment"] = kmeans.fit_predict(customer_df[["Total Sales Scaled"]])

    # Convert to dictionary for HTML rendering
    return render_template("advanced_analysis.html",
                           future_sales=list(zip(range(1, 13), future_sales)),
                           stock_recommendations=stock_recommendations.to_dict(orient="records"),
                           customer_segments=customer_df.to_dict(orient="records"))

@app.route('/download_report')
def download_report():
    file_path = session.get("uploaded_file")
    
    if not file_path or not os.path.exists(file_path):
        flash("No file uploaded! Please upload a file first.", "danger")
        return redirect(url_for("upload_file"))
    
    # Load Data
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Ensure necessary columns exist
    required_columns = {"Product Name", "Customer ID", "Quantity Sold", "Total Sales", "Unit Price", "Category", "Store Location"}
    if not required_columns.issubset(df.columns):
        flash("Missing required columns in the file.", "danger")
        return redirect(url_for("dashboard"))

    # 📌 *Top-Selling Products*
    top_products = df.groupby("Product Name")["Quantity Sold"].sum().reset_index()
    top_products = top_products.sort_values(by="Quantity Sold", ascending=False)

    # 📌 *Top Customers*
    top_customers = df.groupby("Customer ID")["Total Sales"].sum().reset_index()
    top_customers = top_customers.sort_values(by="Total Sales", ascending=False)

    # 📌 *Total Sales & Profit*
    total_sales = df["Total Sales"].sum()
    df["Cost Price"] = df["Unit Price"] * 0.7  # Assuming 70% cost
    df["Profit"] = (df["Unit Price"] - df["Cost Price"]) * df["Quantity Sold"]
    total_profit = df["Profit"].sum()

    # 📌 *Stock Inventory Analysis*
    stock_data = df.groupby(["Product Name", "Category"], as_index=False).agg({
        "Quantity Sold": "sum",
        "Unit Price": "mean"
    })
    stock_data["Stock Value"] = stock_data["Quantity Sold"] * stock_data["Unit Price"]

    # 📌 *Customer Buying Patterns*
    customer_patterns = df.groupby("Customer ID")["Product Name"].apply(list).reset_index()

    # 📌 *Supplier Orders Analysis*
    df["Supplier"] = df["Product Name"].apply(lambda x: f"Supplier {ord(x[0]) % 5 + 1}")  # Generate supplier names
    supplier_performance = df.groupby("Supplier")["Total Sales"].sum().reset_index()
    supplier_performance = supplier_performance.sort_values(by="Total Sales", ascending=False)

    # 📌 *Create an Excel file*
    print(io)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Raw Sales Data", index=False)
        top_products.to_excel(writer, sheet_name="Top Selling Products", index=False)
        top_customers.to_excel(writer, sheet_name="Top Customers", index=False)
        stock_data.to_excel(writer, sheet_name="Stock Inventory", index=False)
        customer_patterns.to_excel(writer, sheet_name="Customer Patterns", index=False)
        supplier_performance.to_excel(writer, sheet_name="Supplier Orders", index=False)

        # 📌 Summary Sheet
        summary_df = pd.DataFrame({
            "Metric": ["Total Sales", "Total Profit"],
            "Value": [total_sales, total_profit]
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)

    return send_file(output, as_attachment=True, download_name="sales_analysis_report.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    app.run(debug=True)
