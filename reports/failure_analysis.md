# Failure Analysis — Lab 19

## Case 1 — GraphRAG advantage
- **Question:** G5000-08 — Which external organizations are connected to ServiceNow's generative-AI efforts in the selected data, and what distinct role does each play?
- **Flat answer:** ServiceNow's generative-AI efforts involve partnerships with two key external organizations, each playing a distinct role:

1. **NVIDIA**: Partnered with ServiceNow to develop powerful enterprise-grade generative AI capabilities aimed at transforming business processes through faster, more intelligent workflow automation [chunk_id=https://www.tmcnet.com/usubmit/2023/05/17/9816504.htm::c0000].

2. **Deloitte**: Expanded its alliance with ServiceNow to integrate Now Assist generative AI capabilities into next-generation managed services. Deloitte focuses on helping customers manage ongoing operational and technology needs strategically, enhancing employee and enterprise experiences [chunk_id=https://www.businesswire.com/news/home/20231025715145/en/Deloitte-and-ServiceNow-Expand-Alliance-to-Integrate-Now-Assist-Generative-AI-Capabilities-With-Next-Generation-Managed-Services::c0000].

No other external organizations connected to ServiceNow's generative AI efforts are mentioned in the provided data.
- **Graph answer:** ServiceNow's generative-AI efforts are connected to the following external organizations, each playing a distinct role:

1. **NVIDIA**: Partnered with ServiceNow to develop powerful enterprise-grade generative AI capabilities aimed at transforming business processes through faster, more intelligent workflow automation [source_row=746].

2. **Accenture**: Partnered alongside ServiceNow and NVIDIA to accelerate generative AI adoption for enterprises, indicating a collaborative role in implementation and scaling of generative AI solutions [source_row=648].

3. **Deloitte**: Expanded its alliance with ServiceNow to integrate Now Assist generative AI capabilities with next-generation managed services, focusing on helping customers manage ongoing operational and technology needs using generative AI [source_row=411].

These partnerships highlight NVIDIA's role in technology development, Accenture's role in enterprise adoption acceleration, and Deloitte's role in integrating generative AI into managed services for operational support.
- **Quality delta:** 3.33
- **Root cause interpretation:** Flat retrieval ranks chunks independently; GraphRAG can connect canonical entities and provenance-bearing edges before combining vector evidence.

## Case 2 — GraphRAG failure / weakest relative case
- **Question:** G5000-44 — What two distinct partner ecosystems connect L&T Technology Services to advanced infrastructure in 2023: one for urban-rail 5G and one for OT security?
- **Flat answer:** In 2023, L&T Technology Services connected to advanced infrastructure through two distinct partner ecosystems: 

1. For urban-rail 5G, L&T Technology Services partnered with Qualcomm and Thales to enable 5G private networks in urban railways [chunk_id=http://www.itnewsonline.com/BusinessWire/LT-Technology-Services-and-Qualcomm-Selected-by-Thales-for-Enabling-5G-Private-Networks-in-Urban-Railways/30401].

2. For OT (Operational Technology) security, L&T Technology Services joined forces with Palo Alto Networks as an MSSP partner to offer comprehensive security platforms for OT and address industrial segments [chunk_id=https://finance.yahoo.com/news/l-t-technology-services-joins-083000328.html::c0000].
- **Graph answer:** In 2023, L&T Technology Services connects to advanced infrastructure through two distinct partner ecosystems: 

1. For urban-rail 5G, the partnership is implied through their acquisition of the Smart World & Communication (SWC) business of Larsen & Toubro Ltd (L&T), which likely supports advanced communication infrastructure [source_row=1007].

2. For OT (Operational Technology) security, L&T Technology Services partnered with Palo Alto Networks as an MSSP (Managed Security Service Provider) partner to offer comprehensive OT security platforms addressing industrial segments [source_row=471].
- **Quality delta:** -3.00
- **Likely root causes to audit:** missing extraction edge, imperfect seed/entity resolution, or super-node recency cap. `graph_route`, matched seeds and edge count are exported per question for tracing rather than guessing.
