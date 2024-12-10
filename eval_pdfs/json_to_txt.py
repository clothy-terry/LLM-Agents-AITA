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


extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs\questions.txt', 'question')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs\ref_rubric.txt', 'referenceAnswer')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs\context.txt', 'context')
extract_field_to_txt('eval_dataset/linear_algebra_benchmark.json', 'eval_pdfs\o1_student_ans.txt', 'o1Answer')

