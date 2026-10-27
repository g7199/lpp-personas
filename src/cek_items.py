"""CEK items, human answer coding, demographic persona sentence and engagement sentences (shared by all CEK scripts)."""
import zlib, numpy as np
DOM = ['climate', 'wall', 'amtrak', 'vets', 'flat']
DOMTXT = {'climate': 'climate change and environmental policy', 'wall': 'regulating banks and Wall Street', 'amtrak': 'government spending on transportation and infrastructure',
          'vets': "how the government provides veterans' care", 'flat': 'taxes and the federal tax system'}
# (item id, domain, question, labels in human order) -- main DVs use the exact CEK wording; sub-items use the CEK endpoints
ITEMS = [
 ('climate_main', 'climate', "Would you say that climate change is best described as a …", ("Crisis", "Not a problem at all")),
 ('climate_2', 'climate', "From what you've read and heard, do you believe increases in Earth's temperature are due…", ("Entirely due to the effects of pollution from human activity", "Entirely due to natural causes")),
 ('climate_3', 'climate', "Will the solution to climate change come from the free market or from government policies?", ("Entirely from the free market", "Entirely from government policies")),
 ('climate_4', 'climate', "Thinking about what's in the news, is the seriousness of global warming generally exaggerated, correct, or underestimated?", ("Generally exaggerated", "Generally underestimated")),
 ('wall_main', 'wall', "How much confidence do you have in Wall Street bankers and brokers to do the right thing...", ("A great deal", "None at all")),
 ('wall_4', 'wall', "Compared to what it's doing now, do you think the federal government needs to regulate banks and financial institutions...", ("A lot more", "A lot less")),
 ('amtrak_main', 'amtrak', "Do you think the government should spend more, less, or about what it does now on transportation and infrastructure?", ["A lot more", "Somewhat more", "A little more", "About what it's doing now", "A little less", "Somewhat less", "A lot less"]),
 ('amtrak_2', 'amtrak', "Would you prefer government pay for building and maintaining roads and infrastructure through raising taxes for transportation spending, or through charging user-fees, like paying tolls when you drive on the highways?", ("Fund entirely through tax increases", "Fund entirely through user fees")),
 ('amtrak_3', 'amtrak', "If new taxes were raised for transportation, should the money go to building NEW projects or to maintaining EXISTING infrastructure?", ("Entirely new projects", "Entirely existing infrastructure")),
 ('vets_main', 'vets', "How much confidence do you have in the Department of Veterans Affairs' ability to care for veterans?", ("A great deal", "None at all")),
 ('vets_3', 'vets', "Do you favor or oppose giving veterans money to buy private health insurance?", ["Strongly favor", "Favor", "Somewhat favor", "Undecided", "Somewhat oppose", "Oppose", "Strongly oppose"]),
 ('flat_main', 'flat', "Would you favor or oppose changing the federal tax system to a flat tax, where everyone making more than $50,000 a year pays the same percentage of his or her income in taxes?", ["Strongly favor", "Favor", "Somewhat favor", "Undecided", "Somewhat oppose", "Oppose", "Strongly oppose"]),
 ('flat_3', 'flat', "Do you favor or oppose reducing the business and corporate tax rate to 14.5% percent?", ["Strongly favor", "Favor", "Somewhat favor", "Undecided", "Somewhat oppose", "Oppose", "Strongly oppose"]),
 ('flat_4', 'flat', "Would a flat tax do more to help ALL Americans or WEALTHY Americans?", ("ALL Americans", "WEALTHY Americans")),
]
COL = {'climate_main': 'dv_climate_main_w1', 'climate_2': 'dv_climate_2_w1', 'climate_3': 'dv_climate_3_w1', 'climate_4': 'dv_climate_4_w1', 'wall_main': 'dv_wall_main_w1', 'wall_4': 'dv_wall_4_w1',
       'amtrak_main': 'dv_amtrak_main_w1', 'amtrak_2': 'dv_amtrak_2_w1', 'amtrak_3': 'dv_amtrak_3_w1', 'vets_main': 'dv_vets_main_w1', 'vets_3': 'dv_vets_3_w1', 'flat_main': 'dv_flat_main_w1', 'flat_3': 'dv_flat_3_w1', 'flat_4': 'dv_flat_4_w1'}
PID = {1: 'a strong Democrat', 2: 'a not very strong Democrat', 3: 'an independent who leans Democratic', 4: 'an independent', 5: 'an independent who leans Republican', 6: 'a not very strong Republican', 7: 'a strong Republican'}
IDEO = {'Liberal': 'liberal', 'Moderate': 'moderate', 'Conservative': 'conservative', 'Libertarian': 'libertarian', 'Other': "haven't thought much about it"}
def demo(r): return f"You are a {r.age_5.replace(' - ', '-')} year-old {r.race_4} {'woman' if r.female == 1 else 'man'} living in the United States. Your highest level of education is {str(r.educ_5).lower()}. Generally speaking, you think of yourself as {PID[int(r.pid_7)]}, and when it comes to politics you think of yourself as {IDEO[r.ideo_5]}."
ENG = {'high': "You follow politics very closely, and your opinions on different issues tend to fit together into a consistent political outlook.",
       'mid': "You follow politics somewhat. Your opinions mostly line up with your general political leanings, but not on every issue.",
       'low': "You rarely follow politics. You form opinions issue by issue, based on your own experiences and gut feelings, and they often do not line up with a single political ideology."}
P_ENG = {'high': 0.25, 'mid': 0.35, 'low': 0.40}
BINS = [(-np.inf, -1.2, 'noticeably more liberal than'), (-1.2, -0.4, 'somewhat more liberal than'), (-0.4, 0.4, 'about the same as'), (0.4, 1.2, 'somewhat more conservative than'), (1.2, np.inf, 'noticeably more conservative than')]
def latents(sid):
    g = np.random.default_rng(zlib.crc32(sid.encode())); e = g.choice(list(P_ENG), p=list(P_ENG.values())); z = g.normal(0, 1, len(DOM)); return e, dict(zip(DOM, z))
def lpp(r, doms):
    e, z = latents(r.subject_id); s = demo(r) + " " + ENG[e]
    dev = [f"- On {DOMTXT[d]}, your views are {next(t for lo, hi, t in BINS if lo <= z[d] < hi)} most people who share your general politics." for d in doms]
    return s + ("\nYour views on specific issues:\n" + "\n".join(dev) if dev else "")
def scale(lab): return "\n".join(f"{i + 1} = {l}" for i, l in enumerate(lab)) if isinstance(lab, list) else f"1 = {lab[0]}\n7 = {lab[1]}"
def joint_prompt(per):
    qs = "\n\n".join(f"Q{i + 1}. {q}\n{scale(lab)}" for i, (_, _, q, lab) in enumerate(ITEMS))
    return per + "\n\nNow answer the following survey questions as this person would.\n\n" + qs + "\n\nAnswer each question with a single number from 1 to 7. Respond with only a JSON object of the form {" + ", ".join(f'"Q{i + 1}": n' for i in range(len(ITEMS))) + "}. Do not run any commands or read any files."
def single_prompt(per, it):
    _, _, q, lab = it; return per + "\n\nNow answer the survey question as this person would.\n" + f"{q}\n{scale(lab)}\nAnswer with a single number from 1 to 7. Respond with only the number. Do not run any commands or read any files."
