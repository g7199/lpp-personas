# Survey data

Download the files below and place them at the paths shown (relative to `data/`, or to `$LPP_DATA`).

**CEK** (Coppock, Ekins and Kirby 2018, op-ed experiments). We use the cleaned MTurk file from the replication archive of
Coppock and Green (2022), "Do Belief Systems Exhibit Dynamic Constraint?", *Journal of Politics* 84(2), JOP Dataverse
(https://dataverse.harvard.edu/dataverse/jop).

    cg/study_1_mturk_cleaned.rds

**GSS** (General Social Survey, NORC, https://gss.norc.org). The cumulative 1972-2024 cross-section file in Stata format
(release 3a), and the 2006-2010 panel (release 6a) for the retest reference.

    gss/GSS_stata/gss7224_r3a.dta
    gsspanel/GSS_panel06w123_R6a - Stata.dta

`src/prepare_gss.py` reduces the cumulative file to the variables used here (`gss/gss_sub.parquet`).

**HM** (Hopkins and Mummolo 2017, single-frame replications on GfK and MTurk), from the archive of Coppock (2023),
*Persuasion in Parallel*, Harvard Dataverse, doi:10.7910/DVN/I9GSKI.

    hm/frame_breath_topic.rds

All three are public, de-identified files. Check each archive's terms before redistributing anything derived from them.
