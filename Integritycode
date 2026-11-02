import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF
import os
import datetime

# ==========================================
# 1. BACKEND ENGINES (Logic)
# ==========================================
class SemanticChecker:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.source_mapping = []

    def build_index(self, other_docs_dict):
        self.source_mapping = []
        for filename, text in other_docs_dict.items():
            sentences = [s.strip() for s in text.split('. ') if len(s) > 15]
            for s in sentences:
                self.source_mapping.append({"filename": filename, "sentence": s})
                
        if not self.source_mapping:
            self.index = None
            return
            
        sentences_only = [item["sentence"] for item in self.source_mapping]
        embeddings = self.model.encode(sentences_only, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1]) 
        self.index.add(embeddings)

    def check_submission(self, student_text, threshold=0.75):
        if not self.index: return []
        student_sentences = [s.strip() for s in student_text.split('. ') if len(s) > 15]
        if not student_sentences: return []
        
        student_embeddings = self.model.encode(student_sentences, convert_to_numpy=True)
        faiss.normalize_L2(student_embeddings)
        scores, indices = self.index.search(student_embeddings, k=1)
        
        flagged = []
        for i, score in enumerate(scores):
            if score[0] > threshold:
                matched_info = self.source_mapping[indices[i][0]]
                flagged.append({
                    "student_text": student_sentences[i],
                    "matched_source": matched_info["sentence"],
                    "source_file": matched_info["filename"],
                    "score": round(float(score[0]) * 100, 1)
                })
        return flagged

class CitationValidator:
    def __init__(self):
        self.headers = {"User-Agent": "IntegrityApp/1.0"}
        self.base_url = "https://api.crossref.org/works"

    def verify_citation(self, title_query):
        try:
            res = requests.get(self.base_url, headers=self.headers, params={"query.title": title_query, "rows": 3, "select": "title"}, timeout=5)
            items = res.json().get("message", {}).get("items", [])
            if not items: return {"status": "flagged"}
            
            returned_title = items[0].get("title", [""])[0]
            if fuzz.ratio(title_query.lower(), returned_title.lower()) > 70:
                return {"status": "verified"}
            return {"status": "flagged"}
        except:
            return {"status": "error"}

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text("text") + "\n" for page in doc)
    
    main_body, bibliography = text, ""
    for kw in ["\nReferences", "\nBibliography", "\nWorks Cited"]:
        if kw in text:
            parts = text.split(kw, 1)
            main_body, bibliography = parts[0], parts[1]
            break
    return main_body, bibliography

# ==========================================
# 2. FRONTEND PIPELINE (4 Pages)
# ==========================================
st.set_page_config(layout="wide", page_title="Integrity Assistant")

@st.cache_resource
def load_models():
    return SemanticChecker(), CitationValidator()

checker, validator = load_models()

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def go_to_page(page_name):
    st.session_state.current_page = page_name

# --- PAGE 1: HOME ---
if st.session_state.current_page == "Home":
    st.title("🛡️ Responsible AI-Based Academic Integrity Assistant")
    st.markdown("This tool helps educators review student submissions for unusual similarities and missing citations.")
    st.button("Upload Assignments ➔", on_click=go_to_page, args=("Upload",), type="primary")

# --- PAGE 2: UPLOAD ---
elif st.session_state.current_page == "Upload":
    st.title("📂 Upload Assignments")
    st.button("⬅ Back to Home", on_click=go_to_page, args=("Home",))
    
    uploaded_files = st.file_uploader("Upload two or more PDFs to compare them", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) > 1:
        if st.button("Analyze Submissions ➔", type="primary"):
            st.session_state.uploaded_files = uploaded_files
            
            # Clear previous analysis if new files are uploaded
            if "analysis_complete" in st.session_state:
                del st.session_state.analysis_complete
                
            go_to_page("Analysis")
    elif uploaded_files and len(uploaded_files) == 1:
        st.warning("Please upload at least TWO files so the system can compare them against each other.")

# --- PAGE 3: ANALYSIS ---
elif st.session_state.current_page == "Analysis":
    st.title("📊 Cross-Analysis Results")
    st.button("⬅ Back to Upload", on_click=go_to_page, args=("Upload",))
    
    with st.spinner("Analyzing and cross-checking documents..."):
        if "analysis_complete" not in st.session_state:
            docs_data = {}
            for uf in st.session_state.uploaded_files:
                temp_path = f"temp_{uf.name}"
                with open(temp_path, "wb") as f:
                    f.write(uf.getvalue())
                main_text, bib_text = process_pdf(temp_path)
                docs_data[uf.name] = {"main": main_text, "bib": bib_text}
                os.remove(temp_path) 
                
            all_sim_results = []
            for current_file, data in docs_data.items():
                other_docs = {name: d["main"] for name, d in docs_data.items() if name != current_file}
                checker.build_index(other_docs)
                
                sim_results = checker.check_submission(data["main"])
                for res in sim_results:
                    res["student_file"] = current_file
                    all_sim_results.append(res)
            
            st.session_state.all_sim_results = all_sim_results
            st.session_state.docs_data = docs_data
            st.session_state.analysis_complete = True
        
        st.markdown(f"### 📄 Total Similarity Overlaps Found: **{len(st.session_state.all_sim_results)}**")
        st.markdown("### Overall Review Status: **Requires Human Review**")
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟡 Peer-to-Peer Similarities")
            if not st.session_state.all_sim_results:
                st.success("No significant semantic overlaps found.")
            
            seen_overlaps = set()
            for res in st.session_state.all_sim_results:
                overlap_pair = tuple(sorted([res['student_file'], res['source_file']]))
                if (overlap_pair, res['student_text']) not in seen_overlaps:
                    seen_overlaps.add((overlap_pair, res['student_text']))
                    
                    with st.expander(f"Similarity: {res['score']}% (Between {res['student_file']} & {res['source_file']})"):
                        st.markdown(f"**Text in {res['student_file']}:**\n> {res['student_text']}")
                        st.markdown(f"**Text in {res['source_file']}:**\n> {res['matched_source']}")

        with col2:
            st.subheader("🔴 Citation Check")
            for filename, data in st.session_state.docs_data.items():
                if not data["bib"].strip():
                    st.error(f"**{filename}**: No bibliography found.")
                else:
                    st.success(f"**{filename}**: Bibliography found and formatted correctly.")
        
    st.divider()
    st.button("View Final Web Report ➔", on_click=go_to_page, args=("Report",), type="primary")

# --- PAGE 4: REPORT (WEB BASED) ---
elif st.session_state.current_page == "Report":
    st.button("⬅ Back to Analysis", on_click=go_to_page, args=("Analysis",))
    
    st.title("📄 Academic Integrity Summary Report")
    st.markdown("---")
    
    # 1. Summary Section
    st.subheader("1. Overview")
    st.write(f"**Total Documents Analyzed:** {len(st.session_state.docs_data)}")
    st.write(f"**Total Overlaps Detected:** {len(st.session_state.all_sim_results)}")
    
    # 2. Detailed Findings
    st.subheader("2. Detected Similarities")
    
    # We will build the text report string dynamically alongside displaying the web report
    txt_report = f"ACADEMIC INTEGRITY SUMMARY REPORT\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    txt_report += "="*50 + "\n\n"
    txt_report += f"Total Documents Analyzed: {len(st.session_state.docs_data)}\n"
    txt_report += f"Total Overlaps Detected: {len(st.session_state.all_sim_results)}\n\n"
    txt_report += "-"*50 + "\nDETECTED SIMILARITIES\n" + "-"*50 + "\n\n"
    
    if not st.session_state.all_sim_results:
        st.success("No overlapping text was found between the submitted documents.")
        txt_report += "No overlapping text was found.\n\n"
    else:
        seen_overlaps = set()
        for res in st.session_state.all_sim_results:
            overlap_pair = tuple(sorted([res['student_file'], res['source_file']]))
            if (overlap_pair, res['student_text']) not in seen_overlaps:
                seen_overlaps.add((overlap_pair, res['student_text']))
                
                # Display on Web
                st.markdown(f"**Match: {res['score']}% between `{res['student_file']}` and `{res['source_file']}`**")
                st.info(f"\"{res['student_text']}\"")
                
                # Add to Text Report
                txt_report += f"MATCH: {res['score']}% between '{res['student_file']}' and '{res['source_file']}'\n"
                txt_report += f"TEXT:\n\"{res['student_text']}\"\n\n"
    
    # 3. Faculty Remarks Section
    st.markdown("---")
    st.subheader("3. Faculty Remarks")
    remarks = st.text_area("Write your final assessment or notes for the students here:", height=150)
    
    # Append remarks to the text report
    txt_report += "-"*50 + "\nFACULTY REMARKS\n" + "-"*50 + "\n"
    txt_report += remarks if remarks.strip() else "No remarks provided."
    txt_report += "\n\n" + "="*50 + "\nEND OF REPORT"
    
    st.markdown("---")
    
    # Action Buttons (Now has both Print to PDF and Download Text file)
    st.markdown("### Actions")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.button("🖨️ Save as PDF (Print)", on_click=lambda: st.write('<script>window.print()</script>', unsafe_allow_html=True), type="primary")
    with col2:
        st.download_button(
            label="⬇️ Download Text Report",
            data=txt_report,
            file_name="Integrity_Report.txt",
            mime="text/plain"
        )
    with col3:
        st.button("🔄 Start New Session", on_click=go_to_page, args=("Home",))
