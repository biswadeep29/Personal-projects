import streamlit as st

# Set the page configuration
st.set_page_config(page_title="My Personal Homepage", page_icon="🌟", layout="wide")

# Title and introduction
st.title("Welcome to My Personal Homepage!")
st.write("Hi! I'm Biswadeep Mazumder. I'm passionate about ML/AI and your mom, and I love working on exciting projects. Explore my work and feel free to reach out!")

# About Section
st.header("About Me")
st.write("""
I'm Biswadeep Mazumder, a 2nd year student JORHAT ENGINEERING COLLEGE.\n
I enjoy riding YOUR MOM. Let's connect to create something amazing!
""")

# Projects Section
st.header("Projects")
st.write("Here are some of the projects I have worked on:")

projects = [
    {"name": "Project 1 : Diamond Price Prediction Model", "description": "A basic model on price prediction using machine learning", "link": "https://i-like-riding-your-mom.streamlit.app/Project_1"},
    {"name": "Project 2", "description": "Description of Project 2", "link": "https://project2-link.com"},
    {"name": "Project 3", "description": "Description of Project 3", "link": "https://project3-link.com"},
]

for project in projects:
    st.subheader(project["name"])
    st.write(project["description"])
    st.markdown(f"[View Project]({project['link']})")

# Contact Section
st.header("Contact Me")
st.write("""
Feel free to get in touch via:
- Email: [biswadeepmazumder29@gmail.com](mailto:biswadeepmazumder29@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/biswadeep-mazumder-4b5819267/](https://www.linkedin.com/in/biswadeep-mazumder-4b5819267/)
""")

# Footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit")
