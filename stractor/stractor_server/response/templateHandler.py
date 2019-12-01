#

from response.requestHandler import RequestHandler


class TemplateHandler(RequestHandler):
	def __init__(self):
		super().__init__()
		self.contentType = 'text/html'

	def find(self, routeData):
		try:
			data = routeData['template']
			template_file = open(f'templates/{data}')
			self.contents = template_file
			self.setStatus(200)
			return True
		except:
			self.setStatus(404)
			return False