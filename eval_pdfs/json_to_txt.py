def extract_field_to_txt(json_file_path, output_txt_path, field_name):
    import json

    # Load the JSON file with specified encoding
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract questions and write them to a new text file
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        for index, item in enumerate(data, start=1):
            question = item.get(field_name, 'No question provided')
            output_file.write(f"{index}. \"{question}\"\n")

    print(f"successfully written to '{output_txt_path}'.")


extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs/l_a_questions.txt', 'question')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs/l_a_ref_rubric.txt', 'referenceAnswer')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs/l_a_context.txt', 'context')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs/l_a_o1_student_ans.txt', 'o1Answer')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs/l_a_rating.txt', 'rating')

extract_field_to_txt('eval_dataset/physics_benchmark.json', 'eval_pdfs/physics_questions.txt', 'Question')
extract_field_to_txt('eval_dataset/physics_benchmark.json', 'eval_pdfs/physics_ref_rubric.txt', 'ReferenceAnswer')
extract_field_to_txt('eval_dataset/physics_benchmark.json', 'eval_pdfs/physics_context.txt', 'Context')
extract_field_to_txt('eval_dataset/physics_benchmark.json', 'eval_pdfs/physics_o1_student_ans.txt', 'o1Answer')
extract_field_to_txt('eval_dataset/physics_benchmark.json', 'eval_pdfs/physics_rating.txt', 'Rating')
# extract_field_to_txt('eval_dataset/QA-manufacturing.json', 'eval_pdfs/qa_questions.txt', 'question')
# extract_field_to_txt('eval_dataset/QA-manufacturing.json', 'eval_pdfs/qa_ref_rubric.txt', 'referenceAnswer')
# extract_field_to_txt('eval_dataset/QA-manufacturing.json', 'eval_pdfs/qa_context.txt', 'context')
# extract_field_to_txt('eval_dataset/QA-manufacturing.json', 'eval_pdfs/qa_o1_student_ans.txt', 'o1Answer')

