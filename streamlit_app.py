import streamlit as st

# Set page config
st.set_page_config(
    page_title="Resume Relief",
    page_icon="assets/logo.jpg",
    layout="wide",
)

# Main app
def main():
    home_page = st.Page("pages/home_page.py", title="Home", icon="🪴")
    page_1 = st.Page("pages/1_jobs_analysis.py", title="Jobs Analysis", icon="✨")
    page_2 = st.Page("pages/2_jobs_recommendation.py", title="Jobs Recommendation", icon="🔎")

    pg = st.navigation([home_page, page_1, page_2])
    pg.run()

    st.markdown("---")
    
    # cols = st.columns([2, 1, 2])
    # with cols[1]:
    #     st.markdown(
    #         """
    #         <div style="text-align: center">
    #             © 2025 Resume Relief<br>
    #             Made by Team 1 in Hsinchu with ❤️<br>
    #         </div>
    #         """, 
    #         unsafe_allow_html=True
    #     )

    cols = st.columns([2, 1, 2])
    with cols[1]:
        st.markdown(
            """
            <div style="text-align: center">
                © 2025 Resume Relief<br>
                Made by <a href="https://renaldi-ega.notion.site" target="_blank">Ren</a> in Hsinchu with ❤️<br>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()