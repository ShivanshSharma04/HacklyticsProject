import taipy as tp
from taipy import Config, Core, Gui
import os


patientHealthViewImage = 'assets\consumerHealthInfoViewing.png'
healthcareGeneralImage = 'assets\what-can-you-do-with-a-master-of-business-administration-in-healthcare.jpeg'
flowchart = 'assets\\flowchartWorkflow.png'

################################################################
#            Configure application                             #
################################################################
def build_message(name):
    return f"Thank you for adding your email to our mailing list {name}!"


# A first data node configuration to model an input name.
input_name_data_node_cfg = Config.configure_data_node(id="input_name")
# A second data node configuration to model the message to display.
message_data_node_cfg = Config.configure_data_node(id="message")
# A task configuration to model the build_message function.
build_msg_task_cfg = Config.configure_task("build_msg", build_message, input_name_data_node_cfg, message_data_node_cfg)
# The scenario configuration represents the whole execution graph.
scenario_cfg = Config.configure_scenario("scenario", task_configs=[build_msg_task_cfg])

################################################################
#            Design graphical interface                        #
################################################################

input_name = "name@domain.com"
message = None


def submit_scenario(state):
    state.scenario.input_name.write(state.input_name)
    state.scenario.submit()
    state.message = scenario.message.read()

# Rename these based on the pages actually needed for the project
page1 = """
# Introduction to **InfoGenesis**{: .color-primary}

<br/>

Our project is focused on using web scraping and NLP techniques to examine patient discussions on health forums, health articles, social media, and patient feedback platforms to understand common concerns, questions, and overall sentiment about particular health conditions or treatments.
We employ generative AI via a fine tuned LLAMA model to create curated health information content and summaries addressing these sentiments. For instance, if there’s a significant concern about the side effects of a new medication, the AI could generate informative content that explains the side effects, ways to mitigate them, and the importance of the medication in treatment.

<br/>

<|{healthcareGeneralImage}|image|{200px}|height|{100px}|width|>

<br/>

The ultimate goal of our project is to facilitate the dissemination of information regarding healthcare concerns (processed from social media) via an LLM the educates patients with summarized healthcare info. 

<br/>

Submit your email to be updated on this project as it develops: <|{input_name}|input|>

<|submit|button|on_action=submit_scenario|>

<|{message}|text|>
"""

page2 = """
# Background
<br/>
**Social Media as a Healthcare Tool and Resource**{: .h3 .color-secondary}
<br/>

In an ever developing society, we’ve seen a rapid increase in the use of social media to help reduce barriers that impede access to healthcare support, knowledge, and resources. 
As of 2024, 4.9 billion people worldwide are now in active use of social media for a variety of purposes, with 90% of US adults using social media to learn about healthcare concerns via sites like YouTube, Facebook, X, Reddit, and other platforms
Social media data provides an untapped wealth of data traditionally inaccessible to the general public. 
Platforms like Reddit host unfiltered, candid discussions about health issues, from minor concerns to major diseases, providing a raw glimpse into public sentiment and experience. 
As such, many individuals have taken to these platforms as a primary source of guidance regarding their critical health concerns. 

<br/>

<|{patientHealthViewImage}|image|>

<br/>
**The Dangers of Misinformation as a Public Health Hazard**{: .h3 .color-primary}
<br/>

Numerous studies have shown there is a serious influx of misinformation spread on social media and healthcare is no exception to this. 
The average person is exposed to a myriad of conflicting anecdotes and biases through the medium and this can often cloud their judgements. 
Moreover, authenticated sources like health journals, research papers, and articles can pose a challenge for the average individual to navigate in search of useful information.

<br/>
**Our Solution**{: .h3 .color-success}
<br/>


Our team has created a product addressing the issue of misinformation on social media and confusion surrounding technical research by creating a user-friendly resource that bridges the gap between scholarly accuracy and social media's accessibility, making trustworthy health information readily available. 
"""

page3 = """
# NLP Analysis

<br/>
**Our Methodology for Analysis**{: .h3}
<br/>

The medium of which we chose to analyze for healthcare concerns in our project was a reddit thread called r/health which features posts regarding recent health news and comments from concerned users.
We were able to scrape this data bank using BeautifulSoup and Selenium for a total data acquisition of 70,000+ words.
Then we utilized natural language processing (NLP) for data processing, sorting conversations into healthcare-related and non-healthcare topics.
This allowed us to isolate over 1,000+ healthcare relevant sentences.
We then employed a classification model to analyze tokenized sentences, allowing us to enhance topic relevancy and accuracy.
Next we trained a LLAMA model using a bank of 12,000 scholarly articles on a variety of health topics.
Using this model, we fed in relevant tags generated by the data processing done on the social media opinions. 
The result is a curated summary about the most pressing health concerns of the general public.

<br/>

<|{flowchart}|image|{200px}|height|{100px}|width|>

<br/>

See our demo and LLAMA model for its output!

"""

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "Intro": page1,
    "Data": page2,
    "Process": page3,
}

if __name__ == "__main__":
    ################################################################
    #            Instantiate and run Core service                  #
    ################################################################
    Core().run()

    ################################################################
    #            Manage scenarios and data nodes                   #
    ################################################################
    scenario = tp.create_scenario(scenario_cfg)

    ################################################################
    #            Instantiate and run Gui service                   #
    ################################################################

    
    
    Gui(pages=pages).run(title="InfoGenesis: The Patient Friendly Healthcare Solution")
