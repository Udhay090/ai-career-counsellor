"""CrewAI orchestration for the AI Career Counselor."""

from crewai import Agent, Crew, Process, Task


def build_crew(llm) -> Crew:
    profile_analyzer = Agent(
        role="Profile Analyzer",
        goal="Extract the user's current skills, education, interests and stated career goal",
        backstory="A career coach who reads a short bio and immediately identifies strengths and direction.",
        llm=llm,
        verbose=False,
    )
    skill_gap_analyzer = Agent(
        role="Skill Gap Analyzer",
        goal="Compare the user's profile against their target career and list missing skills",
        backstory="A technical recruiter who knows exactly what skills each role expects.",
        llm=llm,
        verbose=False,
    )
    course_recommender = Agent(
        role="Course Recommender",
        goal="Map each missing skill to a specific, practical resource",
        backstory="A learning specialist who tracks the best free and paid courses across domains.",
        llm=llm,
        verbose=False,
    )
    career_advisor = Agent(
        role="Career Advisor",
        goal="Turn the profile, gaps and resources into one clear answer to the user's actual question",
        backstory="A senior mentor who converts analysis into a concrete, actionable plan.",
        llm=llm,
        verbose=False,
    )

    profile_task = Task(
        description=(
            "Conversation so far:\n{history}\n\n"
            "Latest user message: {query}\n\n"
            "Summarize the user's current skills, interests, and career goal. "
            "If the latest message modifies an earlier goal (e.g. 'make it low-cost'), "
            "treat it as a constraint on the same goal, not a new one."
        ),
        expected_output="Short summary with sections: Current Skills, Interests, Career Goal, Constraints.",
        agent=profile_analyzer,
    )

    gap_task = Task(
        description="Using the profile summary, list the skills the user is missing for their career goal, ordered by priority.",
        expected_output="Bullet list of missing skills, most important first.",
        agent=skill_gap_analyzer,
        context=[profile_task],
    )

    course_task = Task(
        description=(
            "Using the skill gap list, recommend 1-2 concrete resources (name + platform) per skill. "
            "Respect any constraints from the profile summary (e.g. low-cost -> prefer free resources)."
        ),
        expected_output="Bullet list mapping each skill gap to specific resources.",
        agent=course_recommender,
        context=[profile_task, gap_task],
    )

    advisor_task = Task(
        description=(
            "Combine the profile, skill gaps, and resources into one final answer "
            "to the user's latest message: {query}. Be concise and actionable."
        ),
        expected_output="Final answer with sections: Summary, Skill Gaps, Recommended Resources, Next Steps.",
        agent=career_advisor,
        context=[profile_task, gap_task, course_task],
    )

    return Crew(
        agents=[profile_analyzer, skill_gap_analyzer, course_recommender, career_advisor],
        tasks=[profile_task, gap_task, course_task, advisor_task],
        process=Process.sequential,
        verbose=False,
    )
