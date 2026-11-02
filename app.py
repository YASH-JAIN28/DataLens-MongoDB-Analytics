import hashlib, hmac, os, secrets, uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()
st.set_page_config(page_title="Business Data Analyzer", page_icon=":material/analytics:", layout="wide")

st.markdown("""
<style>
.stApp {background:#f5f7f4;color:#111111}
.block-container {max-width:1440px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid #dfe7e2}
[data-testid="stSidebar"] * {color:#111111 !important}
[data-testid="stAppViewContainer"] * {color:#111111}
[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] {color:#111111 !important}
[data-baseweb="tab"] {color:#111111 !important}
[data-baseweb="tab-highlight"] {background:#0f766e !important}
[data-testid="stForm"] {background:#ffffff;border:1px solid #dfe7e2;border-radius:16px;padding:1.35rem 1.35rem .7rem;box-shadow:0 10px 28px rgba(23,33,43,.06)}
[data-testid="stFormSubmitButton"] button,[data-testid="stButton"] button {min-height:2.75rem;border-radius:10px;font-weight:700;letter-spacing:.01em;transition:all .18s ease;background:#eef1ee !important;color:#111111 !important;border:1px solid #b9c9c4 !important;box-shadow:0 4px 12px rgba(23,33,43,.06)}
[data-testid="stFormSubmitButton"] button p,[data-testid="stButton"] button p {color:#111111 !important}
[data-testid="stFormSubmitButton"] button:hover,[data-testid="stButton"] button:hover {background:#e2e8e4 !important;border-color:#8fa39c !important;transform:translateY(-1px)}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], [data-baseweb="select"] [role="combobox"] {background-color:#f7f8f6 !important;border-color:#b9c9c4 !important}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea,[data-baseweb="select"] input {color:#111111 !important;background-color:#f7f8f6 !important}
[data-testid="stTextInput"] > div > div,[data-testid="stSelectbox"] [data-baseweb="select"] > div {background-color:#f7f8f6 !important;border:1px solid #b9c9c4 !important}
[data-testid="stTextInput"] input,[data-testid="stSelectbox"] [data-baseweb="select"] * {color:#111111 !important}
[data-testid="stFileUploader"] section {background:#f7f8f6 !important;border:1px dashed #9aaea6 !important;border-radius:10px}
[data-testid="stFileUploader"] section * {color:#111111 !important}
[data-testid="stFileUploader"] button {background:#eef1ee !important;color:#111111 !important;border:1px solid #b9c9c4 !important}
[data-testid="stDataFrame"] {border:1px solid #cfdad5 !important;border-radius:10px;overflow:hidden}
[data-testid="stMetric"] {background:#ffffff;border:1px solid #dfe7e2;padding:1rem 1.1rem;border-radius:14px;box-shadow:0 6px 20px rgba(23,33,43,.05)}
[data-testid="stMetricLabel"] {color:#111111;font-weight:600}
[data-testid="stMetricValue"] {color:#17212b;font-family:'Space Grotesk',sans-serif}
.brand-bar {border-bottom:1px solid #dfe7e2;padding:0 0 1.6rem;margin-bottom:1.6rem}
.brand-kicker {color:#0f766e;font-size:.72rem;font-weight:700;letter-spacing:.14em;margin-bottom:.5rem}
.brand-title {color:#17212b;font-family:'Space Grotesk',sans-serif;font-size:2.25rem;font-weight:700;letter-spacing:-.03em;line-height:1.05}
.brand-subtitle {color:#111111;font-size:1rem;margin-top:.55rem}
.auth-shell {max-width:640px;margin:4rem auto 0;background:#ffffff;border:1px solid #dfe7e2;border-radius:20px;padding:2.4rem;box-shadow:0 18px 50px rgba(23,33,43,.09)}
.auth-kicker {color:#0f766e;font-size:.72rem;font-weight:700;letter-spacing:.14em;margin-bottom:.65rem}
.auth-title {color:#17212b;font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;letter-spacing:-.03em}
.auth-copy {color:#111111;margin:.5rem 0 1.4rem}
.section-note {color:#111111;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def connect_db():
    uri=os.getenv("MONGODB_URI","mongodb://localhost:27017").strip()
    try:
        client=MongoClient(uri,serverSelectionTimeoutMS=7000)
        client.admin.command("ping")
        db=client[os.getenv("MONGODB_DB","datalens")]
        db.users.create_index("username", unique=True)
        db.datasets.create_index([("created_at",ASCENDING)])
        db.records.create_index([("user_id",ASCENDING),("dataset_id",ASCENDING)])
        return db,None
    except Exception as e: return None,str(e)

def clean_value(v):
    if pd.isna(v): return None
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return float(v)
    if isinstance(v,np.bool_): return bool(v)
    if isinstance(v,pd.Timestamp): return v.isoformat()
    if isinstance(v,(str,int,float,bool)): return v
    return str(v)

def profile(df):
    return pd.DataFrame([{
        "Column":c,"Type":str(df[c].dtype),"Missing":int(df[c].isna().sum()),
        "Missing %":round(df[c].isna().mean()*100,2),"Unique":int(df[c].nunique())
    } for c in df.columns])

def insights(df):
    out=[f"Dataset has {len(df):,} rows and {len(df.columns)} columns.",
         f"Missing cells: {int(df.isna().sum().sum()):,}.",
         f"Duplicate rows: {int(df.duplicated().sum()):,}."]
    nums=df.select_dtypes(include=np.number)
    if len(nums.columns):
        c=nums.var().idxmax()
        out.append(f"{c} has the highest variance among numeric columns.")
        for col in nums.columns:
            out.append(f"Average {col}: {nums[col].mean():,.2f}.")
            if len(out)>6: break
    cats=df.select_dtypes(include=["object","string","category","bool"])
    if len(cats.columns):
        c=cats.columns[0]
        top=df[c].astype(str).value_counts()
        if len(top): out.append(f"Most frequent {c}: {top.index[0]} ({top.iloc[0]} records).")
    return out

def password_hash(password, salt=None):
    salt=salt or secrets.token_hex(16)
    digest=hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return f"{salt}${digest}"

def password_matches(password, stored):
    try:
        salt,digest=stored.split("$",1)
        candidate=password_hash(password,salt).split("$",1)[1]
        return hmac.compare_digest(candidate,digest)
    except ValueError:
        return False

def auth_screen(db):
    st.markdown("""
        <div class="auth-shell">
            <div class="auth-kicker">BUSINESS INTELLIGENCE WORKSPACE</div>
            <div class="auth-title">Business Data Analyzer</div>
            <div class="auth-copy">Sign in to securely save, understand, and act on your business data.</div>
        </div>
        """, unsafe_allow_html=True)
    login_tab,register_tab=st.tabs(["Login","Create account"])
    with login_tab:
        with st.form("login_form"):
            username=st.text_input("Username")
            password=st.text_input("Password",type="password")
            submitted=st.form_submit_button("Login",type="secondary",width="stretch")
        if submitted:
            user=db.users.find_one({"username":username.strip().lower()})
            if user and password_matches(password,user["password_hash"]):
                st.session_state["user"]={"id":user["user_id"],"username":user["username"]}
                st.rerun()
            st.error("Invalid username or password.")
    with register_tab:
        with st.form("register_form"):
            username=st.text_input("Username",key="register_username")
            password=st.text_input("Password",type="password",key="register_password")
            confirm=st.text_input("Confirm password",type="password")
            submitted=st.form_submit_button("Create account",type="primary",width="stretch")
        if submitted:
            username=username.strip().lower()
            valid_username=username.replace("_"," ").replace("-"," ").isalnum()
            if len(username)<3 or not valid_username:
                st.error("Username must be at least 3 characters and use letters, numbers, - or _.")
            elif len(password)<8:
                st.error("Password must be at least 8 characters.")
            elif password!=confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    user_id=str(uuid.uuid4())
                    db.users.insert_one({"user_id":user_id,"username":username,
                        "password_hash":password_hash(password),"created_at":datetime.now(timezone.utc)})
                    st.session_state["user"]={"id":user_id,"username":username}
                    st.rerun()
                except Exception as e:
                    if "duplicate key" in str(e).lower():
                        st.error("That username is already taken.")
                    else:
                        st.error(f"Could not create account: {e}")

def load_records(db,did,user_id):
    rows=list(db.records.find({"dataset_id":did,"user_id":user_id},
                              {"_id":0,"dataset_id":0,"user_id":0}))
    df=pd.DataFrame(rows)
    for c in df.columns:
        converted=pd.to_numeric(df[c],errors="coerce")
        if df[c].notna().sum() and converted.notna().sum()/df[c].notna().sum()>.85:
            df[c]=converted
    return df

db,error=connect_db()
if db is None:
    st.error("MongoDB connection required: "+error)
    st.markdown("Check that MongoDB is running and that `MONGODB_URI` is correct.")
    st.stop()

if "user" not in st.session_state:
    auth_screen(db)
    st.stop()

user=st.session_state["user"]

st.markdown("""
<div class="brand-bar">
    <div class="brand-kicker">BUSINESS INTELLIGENCE WORKSPACE</div>
    <div class="brand-title">Business Data Analyzer</div>
    <div class="brand-subtitle">Turn sales files into clear decisions with guided analysis, visual trends, and practical insights.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Workspace")
    st.caption(f"Signed in as **{user['username']}**")
    if st.button("Logout",width="stretch"):
        st.session_state.pop("user",None)
        st.session_state.pop("active_id",None)
        st.rerun()
    st.success("MongoDB connected")
    st.subheader("Upload dataset")
    file=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx","xls"])
    custom_name=st.text_input("Dataset name (optional)")
    if file and st.button("Save dataset to MongoDB",type="primary",width="stretch"):
        try:
            raw=pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
            raw.columns=[str(c).strip() or f"column_{i}" for i,c in enumerate(raw.columns)]
            raw=raw.replace([np.inf,-np.inf],np.nan)
            did=str(uuid.uuid4())
            name=custom_name.strip() or file.name
            db.datasets.insert_one({"user_id":user["id"],"dataset_id":did,"dataset_name":name,"filename":file.name,
                "rows":len(raw),"columns":list(raw.columns),"created_at":datetime.now(timezone.utc)})
            batch=[]
            for row in raw.to_dict("records"):
                batch.append({"user_id":user["id"],"dataset_id":did,
                              **{str(k):clean_value(v) for k,v in row.items()}})
                if len(batch)>=500:
                    db.records.insert_many(batch,ordered=False); batch=[]
            if batch: db.records.insert_many(batch,ordered=False)
            st.session_state["active_id"]=did
            st.success(f"Saved {len(raw):,} rows.")
            st.rerun()
        except Exception as e: st.error(f"Upload failed: {e}")

    datasets=list(db.datasets.find({"user_id":user["id"]},{"_id":0}).sort("created_at",-1).limit(100))
if not datasets:
    st.info("Upload a dataset from the sidebar to get started. A sample sales CSV is included in the project folder.")
    st.stop()

labels=[f"{d['dataset_name']} | {d['rows']:,} rows | {d['dataset_id'][:8]}" for d in datasets]
ids=[d["dataset_id"] for d in datasets]
active=st.session_state.get("active_id")
idx=ids.index(active) if active in ids else 0
label=st.selectbox("Choose saved dataset",labels,index=idx)
did=ids[labels.index(label)]
meta=next(d for d in datasets if d["dataset_id"]==did)
df=load_records(db,did,user["id"])
if df.empty:
    st.warning("No rows found in this dataset."); st.stop()

overview,explore,charts,corr,insight=st.tabs(["Overview","Data Explorer","Visualizations","Correlation","Insights & Export"])

with overview:
    st.subheader("Dataset overview")
    a,b,c,d=st.columns(4)
    a.metric("Total rows",f"{len(df):,}")
    b.metric("Columns",len(df.columns))
    c.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows",f"{int(df.duplicated().sum()):,}")
    st.markdown("#### Data preview")
    st.dataframe(df.head(100),width="stretch")
    st.markdown("#### Column profile")
    st.dataframe(profile(df),width="stretch",hide_index=True)
    st.markdown("#### Numeric statistics")
    numeric=df.select_dtypes(include=np.number)
    if len(numeric.columns): st.dataframe(numeric.describe().T.round(2),width="stretch")
    else: st.info("No numeric columns detected.")
    st.markdown("#### Sales mix")
    pie_categories=df.select_dtypes(include=["object","string","category","bool"]).columns.tolist()
    pie_values=numeric.columns.tolist()
    if not pie_categories or not pie_values:
        st.info("Add at least one category column and one numeric sales column to create a pie chart.")
    else:
        pie_category_col,pie_value_col=st.columns(2)
        with pie_category_col:
            pie_category=st.selectbox("Break down sales by",pie_categories,key="pie_category")
        with pie_value_col:
            pie_value=st.selectbox("Measure",pie_values,key="pie_value")
        pie_data=df.groupby(pie_category,dropna=False)[pie_value].sum().reset_index()
        pie_data[pie_category]=pie_data[pie_category].fillna("Unknown").astype(str)
        pie_data=pie_data.sort_values(pie_value,ascending=False).head(12)
        if pie_data[pie_value].sum()>0:
            pie_fig=px.pie(pie_data,names=pie_category,values=pie_value,hole=.46,
                           title=f"{pie_value} distribution by {pie_category}")
            pie_fig.update_traces(textposition="inside",textinfo="percent+label")
            pie_fig.update_layout(template="plotly_white",font=dict(color="#111111"),paper_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(t=60,b=20,l=10,r=10),showlegend=False)
            st.plotly_chart(pie_fig,width="stretch")
        else:
            st.warning("The selected sales column has no positive values for a pie chart.")

with explore:
    st.subheader("Filter and explore records")
    filtered=df.copy()
    chosen=st.multiselect("Select columns to filter",list(df.columns))
    for col in chosen:
        if pd.api.types.is_numeric_dtype(df[col]):
            lo,hi=float(df[col].min()),float(df[col].max())
            if lo<hi:
                val=st.slider(f"{col} range",lo,hi,(lo,hi))
                filtered=filtered[filtered[col].between(*val)]
        else:
            opts=sorted(df[col].dropna().astype(str).unique().tolist())
            vals=st.multiselect(f"{col} values",opts,default=opts[:min(8,len(opts))])
            if vals: filtered=filtered[filtered[col].astype(str).isin(vals)]
    st.caption(f"{len(filtered):,} of {len(df):,} rows")
    st.dataframe(filtered,width="stretch",height=400)
    st.download_button("Download filtered CSV",filtered.to_csv(index=False).encode(), "filtered_data.csv","text/csv")

with charts:
    st.subheader("Create charts")
    allcols=list(df.columns); nums=df.select_dtypes(include=np.number).columns.tolist()
    kind=st.selectbox("Chart type",["Bar","Line","Histogram","Scatter","Box","Pie"])
    x=st.selectbox("X / category",allcols)
    y=st.selectbox("Y / value",nums or allcols)
    color=st.selectbox("Group by (optional)",["None"]+allcols)
    color=None if color=="None" else color
    try:
        if kind=="Bar":
            if x==y:
                tmp=df[x].astype(str).value_counts().head(30).rename_axis(x).reset_index(name="count")
                fig=px.bar(tmp,x=x,y="count",title=f"Count of {x}")
            else:
                tmp=df.groupby(x,dropna=False)[y].mean().reset_index()
                fig=px.bar(tmp,x=x,y=y,color=color if color!=x else None,title=f"Average {y} by {x}")
        elif kind=="Line": fig=px.line(df,x=x,y=y,color=color,title=f"{y} by {x}")
        elif kind=="Histogram": fig=px.histogram(df,x=x,color=color,title=f"Distribution of {x}")
        elif kind=="Scatter":
            if len(nums)<2: st.warning("Scatter plot needs at least two numeric columns."); fig=None
            else:
                y2=st.selectbox("Scatter Y-axis",[n for n in nums if n!=x] or nums)
                fig=px.scatter(df,x=x,y=y2,color=color,title=f"{x} vs {y2}")
        elif kind=="Box": fig=px.box(df,x=x,y=y,color=color,title=f"{y} distribution")
        else:
            tmp=df[x].astype(str).value_counts().head(20).rename_axis(x).reset_index(name="count")
            fig=px.pie(tmp,names=x,values="count",title=f"Distribution of {x}")
        if fig is not None:
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,width="stretch")
    except Exception as e: st.warning(f"Chart could not be created: {e}")

with corr:
    st.subheader("Correlation heatmap")
    numeric=df.select_dtypes(include=np.number)
    if len(numeric.columns)<2: st.info("At least two numeric columns are required.")
    else:
        fig=px.imshow(numeric.corr(),text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",
                      title="Correlation between numeric columns")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch")
        st.caption("Correlation does not prove causation.")

with insight:
    st.subheader("Automatic insights")
    for item in insights(df): st.markdown("- "+item)
    st.markdown("#### Save analysis snapshot")
    if st.button("Save summary to MongoDB"):
        db.analysis_history.insert_one({"user_id":user["id"],"dataset_id":did,"dataset_name":meta["dataset_name"],
            "rows":len(df),"columns":len(df.columns),"missing":int(df.isna().sum().sum()),
            "duplicates":int(df.duplicated().sum()),"created_at":datetime.now(timezone.utc)})
        st.success("Analysis snapshot saved.")
    report="Business Data Analyzer Report\\n"+f"Dataset: {meta['dataset_name']}\\nRows: {len(df)}\\nColumns: {len(df.columns)}\\n\\n"
    report+="COLUMN PROFILE\\n"+profile(df).to_string(index=False)+"\\n\\nINSIGHTS\\n"+"\\n".join(insights(df))
    st.download_button("Download analysis report",report,file_name="business_data_analyzer_report.txt")
    st.download_button("Download full dataset CSV",df.to_csv(index=False).encode(),"dataset.csv","text/csv")

with st.sidebar:
    st.divider()
    if st.button("Delete selected dataset"):
        db.records.delete_many({"user_id":user["id"],"dataset_id":did})
        db.analysis_history.delete_many({"user_id":user["id"],"dataset_id":did})
        db.datasets.delete_one({"user_id":user["id"],"dataset_id":did})
        st.session_state.pop("active_id",None)
        st.rerun()
    st.caption("Delete removes the dataset, its rows and saved analysis snapshots from MongoDB.")
