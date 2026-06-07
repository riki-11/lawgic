# lawgic_tos_topics

This file is the **master reference taxonomy** for all topic classifications the LLM can ever assign to a clause. It is not generated per-document — it is a static, curated file that is injected into every LLM ingestion prompt as a reference guide.

The LLM must only use `topic_id` values that exist in this file. It must never invent new topics.

Suggestions (think about whether or not to incorporate...)
- privacy_related - is this necessary?
- eu_law_relevant - is only relevant for EU consumers. Not for US consumers.
- severity_weight - i get why it is useful but we don't have any ground truth for this.
- tags - where do we get this?
- keywords - where do we get this?
- examples - sure, maybe we can have this.

---

```json
{
  "$schema_version": "1.0.0",
  "topics": [
    {
        "id": "ltd",
        "name": "Limitation of company's liability",
        "parent_topic": "limitation_of_remedy",
        "description": "Clauses concerning limitation of company's liability.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "Unlawful limitation of liablility for damages connected with the use of service, when the company limits its liability for at least one of the following: (a) personal injury or (b) non-performance against consumers or (c) intentionally caused damage"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Lawful limitation - any other than unlawful limitation, for example 'Provided that we have acted with professional diligence, we do not accept responsibility for losses not caused by our breach of these Terms'"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of limitation"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: Unlawful limitation of liablility for damages connected with...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Lawful limitation - any other than unlawful limitation, for...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of limitation]"
            }
        ]
    },
    {
        "id": "ltd_cap",
        "name": "Maximal threshold of company's liability",
        "parent_topic": "limitation_of_remedy",
        "description": "Clauses concerning maximal threshold of company's liability.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "Everyone entitled to the damages connected with using the service may claim them only to a certain amount"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Only Businesses entitled to the damages connected with using the service may claim them only to a certain amount"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "There is no max amount of the damages possible to be claimed"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: Everyone entitled to the damages connected with using the se...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Only Businesses entitled to the damages connected with using...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: There is no max amount of the damages possible to be claimed]"
            }
        ]
    },
    {
        "id": "period",
        "name": "Limitation period",
        "parent_topic": "limitation_of_remedy",
        "description": "Clauses concerning limitation period.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The ToS contains a clause setting the time limit for bringing the action to court, without such ability after this period."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The ToS does not contain a clause described above"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": ""
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The ToS contains a clause setting the time limit for bringin...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The ToS does not contain a clause described above]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: ]"
            }
        ]
    },
    {
        "id": "as_is",
        "name": "No promises",
        "parent_topic": "limitation_of_remedy",
        "description": "Clauses concerning no promises.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "Presence of as-is clause. An \"as-is clause\" is a contractual provision that states that the work or product being provided by the developer or service provider is provided in its current condition, without any additional warranties or guarantees, except for those explicitly stated in the agreement. It serves to limit the developer's liability and makes it clear that the client accepts the work as it is."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": ""
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of as-is clause"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: Presence of as-is clause. An \"as-is clause\" is a contractual...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: ]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of as-is clause]"
            }
        ]
    },
    {
        "id": "indemn",
        "name": "Indemnification clause",
        "parent_topic": "limitation_of_remedy",
        "description": "Clauses concerning indemnification clause.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": ""
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Indemnification clause is present in ToS. An indemnification clause establishes obligation for one party (the indemnifying party) to compensate the other party (the indemnified party) for specific costs and expenses arising from third-party claims or direct claims."
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of indemnification obligation in ToS."
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: ]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Indemnification clause is present in ToS. An indemnification...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of indemnification obligation in ToS.]"
            }
        ]
    },
    {
        "id": "c_law",
        "name": "Choice of law other than user's domicile",
        "parent_topic": "dispute_resolution",
        "description": "Clauses concerning choice of law other than user's domicile.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The ToS provides that a law other than the law of the user's habitual residence will apply to disputes arising in connection with the contract"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The ToS provides that a law other than the law of the user's habitual residence will apply to disputes arising in connection with the contract, but only in relation to businesses"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of choice of law"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The ToS provides that a law other than the law of the user's...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The ToS provides that a law other than the law of the user's...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of choice of law]"
            }
        ]
    },
    {
        "id": "c_forum",
        "name": "Choice of forum other than user's domicile",
        "parent_topic": "dispute_resolution",
        "description": "Clauses concerning choice of forum other than user's domicile.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The ToS provides that disputes arising in connection with the contract shall be resolved in a court of a different jurisdiction from that of the user's permanent residence"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The ToS provides that disputes arising in connection with the contract shall be resolved in a court of a different jurisdiction from that of the user's permanent residence, but only in relation tu businesses"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of choice of forum/\"you can bring claims in your place of domicile\""
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The ToS provides that disputes arising in connection with th...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The ToS provides that disputes arising in connection with th...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of choice of forum/\"you can bring claims in your place...]"
            }
        ]
    },
    {
        "id": "arb",
        "name": "Mandatory arbitration",
        "parent_topic": "dispute_resolution",
        "description": "Clauses concerning mandatory arbitration.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The user is obliged to file a case before an arbitration court specified in the ToS, instead of suing the company with a traditional lawsuit."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The user is obliged to file a case before an arbitration court specified in the ToS, instead of suing the company with a traditional lawsuit, but only for the US citizens and businesses."
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of mandatory arbitration"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The user is obliged to file a case before an arbitration cou...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The user is obliged to file a case before an arbitration cou...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of mandatory arbitration]"
            }
        ]
    },
    {
        "id": "class",
        "name": "Class action waiver",
        "parent_topic": "dispute_resolution",
        "description": "Clauses concerning class action waiver.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The ToS forbids the user to be in a group of people collectively filing a lawsuit as one party in civil proceedings."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The ToS forbids the user, who is a citizen of the US, to be in a group of people collectively filing a lawsuit as one party in civil proceedings."
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of class action waiver"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The ToS forbids the user to be in a group of people collecti...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The ToS forbids the user, who is a citizen of the US, to be...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of class action waiver]"
            }
        ]
    },
    {
        "id": "contr_chg",
        "name": "Unilateral change of contract",
        "parent_topic": "alteration",
        "description": "Clauses concerning unilateral change of contract.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "When the company reserves the right to change the contract without a valid reason (the ToS state the company can always change the contract)"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When the company reserves the right to change the contract for a reason specified in the contract, but the reasons are vague or were not all stated in a contract"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When the company reserves the right to change the contract with a valid reason specified in the contract or does not reserve a right to change it at all"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: When the company reserves the right to change the contract w...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When the company reserves the right to change the contract f...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When the company reserves the right to change the contract w...]"
            }
        ]
    },
    {
        "id": "price_chg",
        "name": "Unilateral change of future prices",
        "parent_topic": "alteration",
        "description": "Clauses concerning unilateral change of future prices.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": ""
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When the company reserves the right to change the future price of services that were not yet supplied or enabled"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When the company does not reserve the right to change the future price of services that were not yet supplied or enabled"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: ]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When the company reserves the right to change the future pri...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When the company does not reserve the right to change the fu...]"
            }
        ]
    },
    {
        "id": "serv_chg",
        "name": "Unilateral change of service by the company",
        "parent_topic": "alteration",
        "description": "Clauses concerning unilateral change of service by the company.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "When the company reserves the right to change the service without a valid reason (when the ToS state, that the company can always change the service or does not state any requriements at all). A service change is a situation, where a company unilaterally modifies the service, by changing existing design, adding or removing features, modyfing purpose of the service, etc."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When the company reserves the right to change the service without a valid reason specified in the contract (the company reserves the right to change the service in a vague way, by using non-precise, general reasons that are hard to interpret by the consumer and give the company freedom to understand them broader than necessary)"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When the company reserves the right to change the service with a valid reason specified in the contract or does not reserve a right to unilaterally change it at all"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: When the company reserves the right to change the service wi...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When the company reserves the right to change the service wi...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When the company reserves the right to change the service wi...]"
            }
        ]
    },
    {
        "id": "acc_del",
        "name": "Account deletion and unilateral termination of contract by the company",
        "parent_topic": "alteration",
        "description": "Clauses concerning account deletion and unilateral termination of contract by the company.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "When the company reserves the right to delete a user’s account without serious grounds or a notice period. By serious grounds we understood situations, where activity of the user was clearly illegal or violated crucial provisions of ToS. The notice period should be reasonably long, to allow the user preparation for termination of contract."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When the company reserves the right to delete a user’s account without serious grounds but with a notice period or with serious grounds but without a notice period."
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When the company reserves the right to delete a user’s account only with serious grounds and a notice period or does not reserve a right to delete it at all"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: When the company reserves the right to delete a user’s accou...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When the company reserves the right to delete a user’s accou...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When the company reserves the right to delete a user’s accou...]"
            }
        ]
    },
    {
        "id": "transfer",
        "name": "Transfer of contractual rights to another subject",
        "parent_topic": "alteration",
        "description": "Clauses concerning transfer of contractual rights to another subject.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "When the ToS allows for contractual rights’ transfer to another subject without user’s consent"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When the ToS allows for contractual rights’ transfer to another subject with user’s consent but only for the company"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When the ToS allows for contractual rights’ transfer with user’s consent and while the consumer is also granted the ability to transfer contractual rights' or the ToS does not contain any provision allowing for the transfer of rights."
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: When the ToS allows for contractual rights’ transfer to anot...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When the ToS allows for contractual rights’ transfer to anot...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When the ToS allows for contractual rights’ transfer with us...]"
            }
        ]
    },
    {
        "id": "cnt_del",
        "name": "User content deletion",
        "parent_topic": "user_policing",
        "description": "Clauses concerning user content deletion.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "When a company reserves a right to delete all content put in the service by the user without restrictions"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When a company reserves a right to delete only the content put in the service by the user that is illegal or violates Terms of Service"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When a company does not reserve a right to delete content put in the service by the user"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: When a company reserves a right to delete all content put in...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When a company reserves a right to delete only the content p...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When a company does not reserve a right to delete content pu...]"
            }
        ]
    },
    {
        "id": "acc_sus",
        "name": "Account suspension",
        "parent_topic": "user_policing",
        "description": "Clauses concerning account suspension.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "When the company reserves the right to suspend a user’s account without serious grounds or a notice period"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "When the company reserves the right to suspend a user’s account without serious grounds but with a notice period or with serious grounds but without a notice period"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "When the company reserves the right to suspend a user’s account only with serious grounds and a notice period or does not reserve a right to suspend it at all"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: When the company reserves the right to suspend a user’s acco...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: When the company reserves the right to suspend a user’s acco...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: When the company reserves the right to suspend a user’s acco...]"
            }
        ]
    },
    {
        "id": "recom",
        "name": "Main parameters used in recommender systems",
        "parent_topic": "regulatory_compliance",
        "description": "Clauses concerning main parameters used in recommender systems.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "Not setting out in the ToS the main parameters used in the fully or partially automated systems used by online platforms to suggest or prioritize information to recipients of the service (recommender system)."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Setting out in the ToS the main parameters used in the recommender system but not in a clear, accessible and easily comprehensible manner"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Setting out in the ToS the main parameters used in the recommender system in compliance with the DSA, which is in a clear, accessible and easily comprehensible manner, as well as any options for the recipients of the service to modify or influence those main parameters that the company may have made available, including at least one option which is not based on profiling"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: Not setting out in the ToS the main parameters used in the f...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Setting out in the ToS the main parameters used in the recom...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Setting out in the ToS the main parameters used in the recom...]"
            }
        ]
    },
    {
        "id": "com_sys",
        "name": "Internal complaint-handling system",
        "parent_topic": "regulatory_compliance",
        "description": "Clauses concerning internal complaint-handling system.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "Lack of internal complaint-handling system that allow the user to file a complain on a decision made by the company concerning user's rights under the contract, before going to court or other institution. Information about a right to present the case to the court or other institution does not count."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Presence of internal complaint-handling system that does not meet all of the requirements stated in article 17 DSA"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Presence of internal complaint-handling system that meets all of the requirements stated in article 17 DSA (available to the user at least 6 months after making the contested decision by the company, electronic, free of charge, easy to access, user-friendly and enabling and facilitating the submission of sufficiently precise and adequately substantiated complaints)"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: Lack of internal complaint-handling system that allow the us...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Presence of internal complaint-handling system that does not...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Presence of internal complaint-handling system that meets al...]"
            }
        ]
    },
    {
        "id": "cnt_retr",
        "name": "Retrieval of digital content by the user",
        "parent_topic": "regulatory_compliance",
        "description": "Clauses concerning retrieval of digital content by the user.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "Lack of clause ensuring the right to retrieve the digital content belonging to the the user after contract's termination."
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Clause ensuring the right to retrieve some of the digital content belonging to the user after contract's termination"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Clause ensuring the right to retrieve all of the digital content belonging to the user after contract's termination"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: Lack of clause ensuring the right to retrieve the digital co...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Clause ensuring the right to retrieve some of the digital co...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Clause ensuring the right to retrieve all of the digital con...]"
            }
        ]
    },
    {
        "id": "IP",
        "name": "Excessive user content IP license",
        "parent_topic": "various",
        "description": "Clauses concerning excessive user content ip license.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "ToS contains an IP license to the content put in the service by the user that neither states explicitly that it is needed to perform the service nor explains other purposes for using user's content"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "ToS contains an IP license to the content put in the service by the user that does not state explicitly that it’s needed to perform the service, but the purposes are listed and explained; or when there is a doubt as to these purposes are necessary"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "ToS contains an IP license to the content put in the service by the user that it states it is needed solely to perform the service or ToS lacks any IP license requirements"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: ToS contains an IP license to the content put in the service...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: ToS contains an IP license to the content put in the service...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: ToS contains an IP license to the content put in the service...]"
            }
        ]
    },
    {
        "id": "discret",
        "name": "Discretional power to interpret the ToS",
        "parent_topic": "various",
        "description": "Clauses concerning discretional power to interpret the tos.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The company reserves the exclusive right to interpret any term of the contract only by itself"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "Lack of discretional power clauses"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": ""
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The company reserves the exclusive right to interpret any te...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: Lack of discretional power clauses]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: ]"
            }
        ]
    },
    {
        "id": "interpret",
        "name": "General interpretation clause",
        "parent_topic": "various",
        "description": "Clauses concerning general interpretation clause.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": "The ToS contains clauses stating that contract must be interpreted in favor of the company's intent"
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The ToS contains clauses stating that contract must be interpreted in in favor of both parties' intent"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "The ToS contains clauses stating that contract must be interpreted in favor of the consumer"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: The ToS contains clauses stating that contract must be inter...]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The ToS contains clauses stating that contract must be inter...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: The ToS contains clauses stating that contract must be inter...]"
            }
        ]
    },
    {
        "id": "sever",
        "name": "Severability clauses",
        "parent_topic": "various",
        "description": "Clauses concerning severability clauses.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": ""
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "The ToS contains provisions that keep the remaining part of the contract in force in case a court declare one or more of its provisions unconstitutional, void, or unenforceable (severability clauses)"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "Lack of severability clauses"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: ]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: The ToS contains provisions that keep the remaining part of...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: Lack of severability clauses]"
            }
        ]
    },
    {
        "id": "suggest",
        "name": "Right to incorporate user's feedback or suggestions without compensation",
        "parent_topic": "various",
        "description": "Clauses concerning right to incorporate user's feedback or suggestions without compensation.",
        "scores": [
            {
                "num_score": -1,
                "label": "bad",
                "explanation": ""
            },
            {
                "num_score": 0,
                "label": "neutral",
                "explanation": "According to ToS, the company has a right to incorporate into the service user feedback or suggestions without compensation"
            },
            {
                "num_score": 1,
                "label": "good",
                "explanation": "According to ToS, the company does not have a right to incorporate into the service user feedback or suggestions without compensation"
            }
        ],
        "examples": [
            {
                "score": -1,
                "clause": "[Example clause reflecting: ]"
            },
            {
                "score": 0,
                "clause": "[Example clause reflecting: According to ToS, the company has a right to incorporate int...]"
            },
            {
                "score": 1,
                "clause": "[Example clause reflecting: According to ToS, the company does not have a right to incor...]"
            }
        ]
    }
  ],
  "parent_topics": [
    {
        "id": "remedy_limit",
        "name": "Limitation of remedy clauses"
    },
    {
        "id": "alteration",
        "name": "Unilateral Alteration",
    },
    {
        "id": "user_policing",
        "name": "Right to police the behaviour of users"
    },
    {
        "id": "dispute_resolution",
        "name": "Dispute Resolution Clauses"
    },
    {
        "id": "reg_reqs",
        "name": "Regulatory requirements"
    },
    {
        "id": "various",
        "name": "various"
    }
  ]
}
```

---

## Field Reference

### Per Topic

| Field | Type | Description |
| --- | --- | --- |
| `id` | `string` | Stable, unique slug. The LLM must use this exact value in `topic_ids` and `primary_topic_id`. |
| `name` | `string` | Human-readable topic name. Used in the UI. |
| `parent_topic` | `string` | ID of the parent group. Used for grouping/clustering in React Flow. |
| `description` | `string` | Definition of the topic's scope. Injected into the LLM prompt. |
| `privacy_related` | `boolean` | Whether this topic involves personal data. Used for filtering in UI. |
| `eu_law_relevant` | `boolean` | Whether this topic has specific grounding in EU Consumer Law. |
| `severity_weight` | `number` | `0.0`–`1.0`. Used to weight this topic's score in the service-level `overall_risk_score` calculation. `1.0` = maximum weight. |
| `tags` | `string[]` | Free-form tags for filtering, search, and UI badge display. |
| `keywords` | `string[]` | Terms the LLM and retrieval pipeline use to route clauses to this topic. Also useful for keyword-based pre-filtering before LLM classification. |
| `scores[]` | `array` | The three scoring tiers: `-1`, `0`, `1`. |
| `scores[].num_score` | `number` | `-1`, `0`, or `1`. |
| `scores[].label` | `string` | `bad`, `neutral`, or `good`. |
| `scores[].explanation` | `string` | Topic-specific criteria for this score. Injected into the LLM prompt. |
| `examples[]` | `array` | Few-shot examples per score level. Injected into the LLM prompt to ensure classification consistency. |
| `examples[].score` | `number` | Score for this example. |
| `examples[].clause` | `string` | Representative example clause for this score. |

---

## Notes for LLM Prompt Injection

When building the ingestion prompt, inject topic data in this order per topic:

1. `name` and `description` — what the topic covers
2. `scores[].explanation` for each score — the scoring criteria
3. `examples` — concrete few-shot examples

The LLM must only ever use `id` values present in this file. If a clause does not match any topic, it should not be included in `notable_clauses`.