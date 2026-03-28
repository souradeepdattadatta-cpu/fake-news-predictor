The Core Issue

The biggest problem right now isn't just that fake news exists—it's that it spreads way faster than the truth can keep up with. Social media has made it so easy for anyone to post a "breaking" headline that looks real but is actually total nonsense. Most people don't have the time or the tools to fact-check every single thing they read, which leads to a lot of unnecessary panic and confusion.
Why Old Solutions Aren't Cutting It

I noticed that most existing tools rely on "blacklists" of known fake websites. The issue there is that hackers and trolls create new domains every single day. By the time a site gets blacklisted, the damage is already done. We can't just keep a list of "bad" sites anymore; we need a system that actually understands the language of misinformation.
The Goal of This Project

My objective was to build a classifier that doesn't just look at where the news came from, but focuses on how it is written. Fake news often uses specific linguistic triggers—like excessive capitalization, sensationalist "clickbait" phrasing, and a lack of neutral reporting.
Technical Objectives

    Pattern Recognition: I wanted to use TF-IDF to identify the specific vocabulary that separates factual reporting from fabricated stories.

    Speed and Scalability: The system needs to work in real-time. If a user pastes a headline into my Streamlit dashboard, they shouldn't have to wait more than a second for an answer.

    Reducing Bias: A major part of my work was cleaning the data so the AI wouldn't just learn that "Reuters = Real." I wanted the model to actually learn the difference in writing styles.

Expected Outcome

By the end of this project, I aimed to provide a reliable, high-accuracy tool (97.59% accuracy) that gives users a "Safety Score" for any piece of text. This helps bridge the gap between human intuition and machine-speed verification, making the internet a slightly safer place to browse.
